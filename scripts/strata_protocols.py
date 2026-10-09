"""Minimal tool roundtrips and strict streaming termination/usage checks."""

import json


def usage_ok(value, input_key="input_tokens", output_key="output_tokens"):
    return isinstance(value, dict) and all(
        type(value.get(k)) is int and value[k] >= 0 for k in (input_key, output_key)
    )


def json_request(runner, path, payload):
    with runner.http(path, payload) as response:
        return json.load(response)


def stream_request(runner, path, payload):
    events = []
    with runner.http(path, dict(payload, stream=True)) as response:
        for line in response:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                events.append(json.loads(line[6:]))
    return events


def qualify(runner, protocol):
    path = "/v1/" + protocol
    row = {"case": protocol + "-tool-stream", "pass": False}
    prompt = (
        "Use lookup for item alpha. After its result, return only its"
        " integer value plus 25."
    )
    function = {
        "name": "lookup",
        "description": "Look up an integer.",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
        },
    }
    try:
        if protocol == "responses":
            user = {"role": "user", "content": prompt}
            payload = {
                "model": "flash-next-orca-iq3xxs",
                "input": [user],
                "max_output_tokens": 256,
                "tools": [{"type": "function", **function}],
                "reasoning": {"effort": "none"},
            }
            first = json_request(runner, path, payload)
            row["first"] = first
            calls = [
                x for x in first.get("output", []) if x.get("type") == "function_call"
            ]
            if len(calls) != 1:
                raise ValueError("Expected one function call")
            call = calls[0]
            if (
                not call.get("call_id")
                or call.get("name") != "lookup"
                or json.loads(call["arguments"]) != {"item": "alpha"}
            ):
                raise ValueError("Invalid tool identity or arguments")
            payload["input"] = [
                user,
                *first["output"],
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "17",
                },
            ]
            events = stream_request(runner, path, payload)
            row["events"] = events
            done = [
                e["response"] for e in events if e.get("type") == "response.completed"
            ]
            if len(done) != 1 or events[-1].get("type") != "response.completed":
                raise ValueError("Missing final completion")
            final = done[0]
            text = "".join(
                c.get("text", "")
                for x in final.get("output", [])
                if x.get("type") == "message"
                for c in x.get("content", [])
            )
            row["pass"] = (
                text.strip() == "42"
                and final.get("status") == "completed"
                and usage_ok(first.get("usage"))
                and usage_ok(final.get("usage"))
            )
        else:
            user = {"role": "user", "content": prompt}
            payload = {
                "model": "flash-next-orca-iq3xxs",
                "messages": [user],
                "max_tokens": 256,
                "thinking": {"type": "disabled"},
                "tools": [
                    {
                        "name": function["name"],
                        "description": function["description"],
                        "input_schema": function["parameters"],
                    }
                ],
            }
            first = json_request(runner, path, payload)
            row["first"] = first
            calls = [x for x in first.get("content", []) if x.get("type") == "tool_use"]
            if first.get("stop_reason") != "tool_use" or len(calls) != 1:
                raise ValueError("Expected tool_use stop and one call")
            call = calls[0]
            if (
                not call.get("id")
                or call.get("name") != "lookup"
                or call.get("input") != {"item": "alpha"}
            ):
                raise ValueError("Invalid tool identity or arguments")
            payload["messages"] = [
                user,
                {"role": "assistant", "content": first["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": "17",
                        }
                    ],
                },
            ]
            events = stream_request(runner, path, payload)
            row["events"] = events
            starts = [e["message"] for e in events if e.get("type") == "message_start"]
            deltas = [e for e in events if e.get("type") == "message_delta"]
            text = "".join(
                e.get("delta", {}).get("text", "")
                for e in events
                if e.get("type") == "content_block_delta"
            )
            final_usage = {
                **(starts[0].get("usage", {}) if starts else {}),
                **(deltas[-1].get("usage", {}) if deltas else {}),
            }
            row["pass"] = (
                text.strip() == "42"
                and len(starts) == 1
                and bool(deltas)
                and deltas[-1].get("delta", {}).get("stop_reason") == "end_turn"
                and bool(events)
                and events[-1].get("type") == "message_stop"
                and usage_ok(first.get("usage"))
                and usage_ok(final_usage)
            )
        row["status"] = 200
    except Exception as e:
        row.update(status="error", error=type(e).__name__)
    runner.save(row)
    return row
