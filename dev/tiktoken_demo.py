"""
Small demo for testing raw tiktoken encode/decode round trips.

Examples:
    uv run python dev/tiktoken_demo.py
    uv run python dev/tiktoken_demo.py --encoding gpt2
    uv run python dev/tiktoken_demo.py --model gpt-4
    uv run python dev/tiktoken_demo.py --check-model gpt-5.4
    uv run python dev/tiktoken_demo.py --list-models
    uv run python dev/tiktoken_demo.py --list-encodings
"""

from __future__ import annotations

import argparse

import tiktoken
from tiktoken.model import MODEL_PREFIX_TO_ENCODING, MODEL_TO_ENCODING


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo raw tiktoken encode/decode.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--encoding",
        default=None,
        help="tiktoken encoding name, e.g. cl100k_base, gpt2, or o200k_base",
    )
    group.add_argument(
        "--model",
        default=None,
        help="OpenAI model name resolved via tiktoken.encoding_for_model(), e.g. gpt-4 or gpt-4o",
    )
    group.add_argument(
        "--check-model",
        default=None,
        help="Check whether tiktoken.encoding_for_model() recognizes a specific model name",
    )
    group.add_argument(
        "--list-models",
        action="store_true",
        help="List GPT-family exact model names and prefixes recognized by the installed tiktoken",
    )
    group.add_argument(
        "--list-encodings",
        action="store_true",
        help="List encoding names available from the installed tiktoken package",
    )
    return parser


def resolve_tokenizer(args: argparse.Namespace) -> tuple[tiktoken.Encoding, str, str]:
    if args.model:
        try:
            enc = tiktoken.encoding_for_model(args.model)
        except Exception as exc:
            raise SystemExit(
                f"failed to resolve model {args.model!r} with tiktoken.encoding_for_model(): {exc}"
            ) from exc
        return enc, "model", args.model

    encoding_name = args.encoding or "cl100k_base"
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        raise SystemExit(
            f"failed to load encoding {encoding_name!r} with tiktoken.get_encoding(): {exc}"
        ) from exc
    return enc, "encoding", encoding_name


def print_supported_models() -> None:
    exact_models = sorted(
        key for key in MODEL_TO_ENCODING if key.startswith("gpt") or key == "gpt2"
    )
    prefix_models = sorted(key for key in MODEL_PREFIX_TO_ENCODING if key.startswith("gpt"))

    print("exact_models:")
    for name in exact_models:
        print(f"  {name} -> {MODEL_TO_ENCODING[name]}")

    print()
    print("prefix_models:")
    for prefix in prefix_models:
        print(f"  {prefix}* -> {MODEL_PREFIX_TO_ENCODING[prefix]}")


def print_supported_encodings() -> None:
    print("encodings:")
    for name in sorted(tiktoken.list_encoding_names()):
        print(f"  {name}")


def check_model(model_name: str) -> None:
    print(f"model: {model_name}")
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception as exc:
        print("recognized: False")
        print(f"error: {type(exc).__name__}: {exc}")
        return

    print("recognized: True")
    print(f"resolved_encoding: {enc.name}")
    print(f"vocab_size: {enc.n_vocab}")


def main() -> None:
    args = build_parser().parse_args()

    if args.list_models:
        print_supported_models()
        return

    if args.list_encodings:
        print_supported_encodings()
        return

    if args.check_model:
        check_model(args.check_model)
        return

    enc, source_type, source_name = resolve_tokenizer(args)

    samples = [
        "Hello, world!",
        "The capital of France is Paris.",
        "Mixing English, numbers 12345, and symbols <>[]{}.",
        "中文测试，看看编码和解码是否一致。",
        "def add(a, b):\n    return a + b\n",
    ]

    print(f"source_type: {source_type}")
    print(f"source_name: {source_name}")
    print(f"resolved_encoding: {enc.name}")
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
