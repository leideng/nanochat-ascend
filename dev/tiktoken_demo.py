"""
Small demo for testing raw tiktoken encode/decode round trips.

Examples:
    uv run python dev/tiktoken_demo.py
    uv run python dev/tiktoken_demo.py --encoding gpt2
"""

from __future__ import annotations

import argparse

import tiktoken


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo raw tiktoken encode/decode.")
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding name, e.g. cl100k_base or gpt2",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    enc = tiktoken.get_encoding(args.encoding)

    samples = [
        "Hello, world!",
        "The capital of France is Paris.",
        "Mixing English, numbers 12345, and symbols <>[]{}.",
        "中文测试，看看编码和解码是否一致。",
        "def add(a, b):\n    return a + b\n",
    ]

    print(f"encoding: {args.encoding}")
    print(f"vocab_size: {enc.n_vocab}")
    print()

    for index, text in enumerate(samples, start=1):
        token_ids = enc.encode(text)
        decoded = enc.decode(token_ids)
        ok = decoded == text

        print(f"sample {index}:")
        print(f"text: {text!r}")
        print(f"token_count: {len(token_ids)}")
        print(f"token_ids: {token_ids}")
        print(f"decoded: {decoded!r}")
        print(f"round_trip_ok: {ok}")
        print()

        if not ok:
            raise SystemExit(f"round-trip mismatch for sample {index}")

    print("all samples passed")


if __name__ == "__main__":
    main()
