#!/usr/bin/env python3
"""Smoke-test text and forced tools across vLLM's three agent-facing APIs."""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


def post(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def validate_model(base_url: str, model: str) -> dict[str, Any]:
    chat = post(
        base_url,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply exactly: api-ok"}],
            "max_tokens": 16,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    responses = post(
        base_url,
        "/v1/responses",
        {
            "model": model,
            "input": "Reply exactly: responses-ok",
            "max_output_tokens": 16,
            "temperature": 0,
        },
    )
    messages = post(
        base_url,
        "/v1/messages",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply exactly: messages-ok"}],
            "max_tokens": 16,
            "temperature": 0,
        },
    )

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    chat_tool = post(
        base_url,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Use the weather tool for Singapore."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather lookup",
                        "parameters": schema,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 128,
            "temperature": 0,
        },
    )

    chat_text = chat["choices"][0]["message"]["content"]
    message_text = "".join(
        item.get("text", "")
        for item in messages.get("content", [])
        if item.get("type") == "text"
    )
    response_output = responses.get("output", [])
    response_types = [item.get("type") for item in response_output]
    response_text = "".join(
        content.get("text", "")
        for item in response_output
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    )
    tool_calls = chat_tool["choices"][0]["message"].get("tool_calls", [])

    if chat_text.strip() != "api-ok":
        raise RuntimeError(f"{model}: unexpected chat response {chat_text!r}")
    if message_text.strip() != "messages-ok":
        raise RuntimeError(f"{model}: unexpected Messages response {message_text!r}")
    if response_text.strip() != "responses-ok":
        raise RuntimeError(f"{model}: unexpected Responses output {response_text!r}")
    if not tool_calls or tool_calls[0]["function"]["name"] != "get_weather":
        raise RuntimeError(f"{model}: forced tool call failed")
    arguments = tool_calls[0]["function"].get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if arguments != {"city": "Singapore"}:
        raise RuntimeError(f"{model}: unexpected tool arguments {arguments!r}")

    return {
        "model": model,
        "chat": chat_text,
        "messages": message_text,
        "responses_output_types": response_types,
        "responses": response_text,
        "forced_tool": tool_calls[0]["function"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model ID to validate; repeat for multiple IDs",
    )
    parser.add_argument("--base-model")
    parser.add_argument("--adapter-model")
    args = parser.parse_args()
    models = [*args.model]
    models.extend(
        model for model in (args.base_model, args.adapter_model) if model is not None
    )
    models = list(dict.fromkeys(models))
    if not models:
        parser.error("provide --model or the legacy --base-model/--adapter-model")
    result = [validate_model(args.base_url, model) for model in models]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
