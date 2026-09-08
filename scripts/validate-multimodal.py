#!/usr/bin/env python3
"""Validate image inputs and public multimodal guardrails."""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post(
    url: str, payload: dict[str, Any], headers: dict[str, str]
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            body = json.load(error)
        except Exception:
            body = {"error": {"message": "non-JSON error"}}
        return error.code, body


def output_text(response: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in response.get("output", [])
        for part in item.get("content", [])
    )


def message_text(response: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for part in response.get("content", [])
        if part.get("type") == "text"
    )


def require_ocr(
    status: int,
    response: dict[str, Any],
    model: str,
    text: str,
    protocol: str,
) -> None:
    if status != 200 or response.get("model") != model or "7429" not in text:
        raise RuntimeError(f"{protocol} failed for {model}: status={status}")
    print(f"ok: {model} {protocol} image")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3.8-27b", "qwen3.8-27b-uncensored"],
    )
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="target a vLLM server directly (/v1/... paths, no gateway key) "
        "instead of the public gateway (/openai/..., /api/v1/messages)",
    )
    parser.add_argument(
        "--remote-url-probe",
        default="http://127.0.0.1:18103/health",
        help="HTTP media URL that must be rejected by the server",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.direct:
        raise SystemExit(f"missing {args.api_key_env}")
    image_bytes = args.image.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode()
    image_url = f"data:image/png;base64,{image_b64}"
    base = args.base_url.rstrip("/")
    if args.direct:
        chat_path, responses_path, messages_path = (
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
        )
    else:
        chat_path, responses_path, messages_path = (
            "/openai/chat/completions",
            "/openai/responses",
            "/api/v1/messages",
        )
    bearer = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    anthropic_headers = {"anthropic-version": "2023-06-01"}
    if api_key:
        anthropic_headers["x-api-key"] = api_key
    instruction = "Read the four digits in this image. Reply with the digits only."

    for model in args.models:
        status, response = post(
            f"{base}{chat_path}",
            {
                "model": model,
                "max_tokens": 32,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
            },
            bearer,
        )
        chat_text = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        require_ocr(status, response, model, chat_text, "chat")

        status, response = post(
            f"{base}{responses_path}",
            {
                "model": model,
                "max_output_tokens": 32,
                "temperature": 0,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": instruction},
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": image_url,
                            },
                        ],
                    }
                ],
            },
            bearer,
        )
        require_ocr(status, response, model, output_text(response), "responses")

        status, response = post(
            f"{base}{messages_path}",
            {
                "model": model,
                "max_tokens": 32,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": instruction},
                        ],
                    }
                ],
            },
            anthropic_headers,
        )
        require_ocr(status, response, model, message_text(response), "messages")

    model = args.models[0]
    status, _ = post(
        f"{base}{chat_path}",
        {
            "model": model,
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe."},
                        {
                            "type": "image_url",
                            "image_url": {"url": args.remote_url_probe},
                        },
                    ],
                }
            ],
        },
        bearer,
    )
    if status not in {400, 403, 422}:
        raise RuntimeError(f"remote media URL was not rejected: status={status}")
    print(f"ok: remote media URL denied ({status})")

    images = [{"type": "image_url", "image_url": {"url": image_url}} for _ in range(5)]
    status, _ = post(
        f"{base}{chat_path}",
        {
            "model": model,
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe."}, *images],
                }
            ],
        },
        bearer,
    )
    if status not in {400, 403, 422}:
        raise RuntimeError(f"five-image request was not rejected: status={status}")
    print(f"ok: image-count limit enforced ({status})")


if __name__ == "__main__":
    main()
