#!/usr/bin/env python3
"""Minimal OpenRouter max_tokens debugger."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def _resolve_api_key_with_source() -> tuple[str, str]:
    for key_name in ("OPENROUTER_API_KEY", "APIKEY_OPENROUTER", "OPENROUTER_KEY"):
        value = os.getenv(key_name, "").strip()
        if value:
            return value, key_name
    raise RuntimeError(
        "OpenRouter API key not found. Set OPENROUTER_API_KEY (or APIKEY_OPENROUTER / OPENROUTER_KEY)."
    )


def _mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 10:
        return f"{value[:2]}...{value[-2:]}"
    return f"{value[:6]}...{value[-4:]}"


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise ValueError(f"max_tokens values must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("No valid max_tokens values were provided.")
    return values


def _extract_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _classify_error(exc: Exception) -> str:
    text = str(exc)
    lower = text.lower()

    if "403" in lower and "key limit exceeded" in lower:
        return "quota_exceeded"
    if "429" in lower or "rate limit" in lower:
        return "rate_limited"
    if "401" in lower or "unauthorized" in lower or "invalid api key" in lower:
        return "auth_error"
    if "400" in lower and "max_tokens" in lower:
        return "invalid_max_tokens"
    return "request_error"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one OpenRouter model with different max_tokens values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=(os.getenv("REPTRACE_OPENROUTER_TEST_MODEL", "").strip() or "openai/gpt-4o-mini"),
        help="OpenRouter model id.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: ACCESS_OK",
        help="Prompt to send for each trial.",
    )
    parser.add_argument(
        "--max-tokens-list",
        default="16,32,64,128,256,512,1024",
        help="Comma-separated max_tokens values to test.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print key source and request settings.",
    )
    parser.add_argument("--http-referer", default=os.getenv("OPENROUTER_HTTP_REFERER", "").strip())
    parser.add_argument("--x-title", default=os.getenv("OPENROUTER_X_TITLE", "").strip())
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print full response text for each trial (instead of a preview).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(override=False)
    args = parse_args()

    try:
        max_tokens_values = _parse_int_list(args.max_tokens_list)
        api_key, key_source = _resolve_api_key_with_source()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=args.timeout,
    )

    extra_headers: dict[str, str] = {}
    if args.http_referer:
        extra_headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        extra_headers["X-Title"] = args.x_title

    print(f"model={args.model}")
    print(f"prompt={args.prompt}")
    print(f"max_tokens_list={max_tokens_values}")
    if args.verbose:
        print(f"key_source={key_source}")
        print(f"key_masked={_mask_secret(api_key)}")
        print(f"timeout={args.timeout}")
        print(f"temperature={args.temperature}")
        print(f"top_p={args.top_p}")

    print("-" * 90)

    for max_tokens in max_tokens_values:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                max_tokens=max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_headers=extra_headers or None,
            )

            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            text = _extract_text(choice.message)
            usage = getattr(response, "usage", None)

            usage_prompt = getattr(usage, "prompt_tokens", None) if usage else None
            usage_completion = getattr(usage, "completion_tokens", None) if usage else None
            usage_total = getattr(usage, "total_tokens", None) if usage else None

            summary = {
                "max_tokens": max_tokens,
                "finish_reason": finish_reason,
                "has_content": bool(text),
                "content_len": len(text),
                "usage_prompt_tokens": usage_prompt,
                "usage_completion_tokens": usage_completion,
                "usage_total_tokens": usage_total,
            }
            print(json.dumps(summary, ensure_ascii=False))

            if args.show_text:
                print(text if text else "<empty>")
            else:
                preview = text[:160].replace("\n", "\\n") if text else "<empty>"
                print(f"preview={preview}")

        except Exception as exc:
            error_type = _classify_error(exc)
            error_summary = {
                "max_tokens": max_tokens,
                "error_type": error_type,
                "error": str(exc),
            }
            print(json.dumps(error_summary, ensure_ascii=False))

        print("-" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
