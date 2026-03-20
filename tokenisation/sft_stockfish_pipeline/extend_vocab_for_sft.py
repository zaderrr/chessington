#!/usr/bin/env python3
"""Extend an existing chess vocab with SFT control tokens."""

import argparse
import json
from pathlib import Path


DEFAULT_TOKENS = ["<STM_WIN>", "<STM_DRAW>", "<STM_LOSS>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add SFT tokens to existing vocab JSON"
    )
    parser.add_argument("--vocab", required=True, help="Input vocab JSON path")
    parser.add_argument("--output", required=True, help="Output vocab JSON path")
    parser.add_argument(
        "--tokens",
        default=",".join(DEFAULT_TOKENS),
        help="Comma-separated tokens to add",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab_path = Path(args.vocab)
    output_path = Path(args.output)

    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    token_to_id = data.get("token_to_id", {})
    if not isinstance(token_to_id, dict):
        raise ValueError("Invalid vocab format: token_to_id missing or not a dict")

    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    added = 0
    for token in tokens:
        if token in token_to_id:
            continue
        token_to_id[token] = len(token_to_id)
        added += 1

    data["size"] = len(token_to_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Done. added={added} size={len(token_to_id)} output={output_path}")


if __name__ == "__main__":
    main()
