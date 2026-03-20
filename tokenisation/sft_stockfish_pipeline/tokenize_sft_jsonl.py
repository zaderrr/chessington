#!/usr/bin/env python3
"""Tokenize Stockfish SFT JSONL into flat ids + loss mask binaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


MOVE_TOKEN_PATTERN = re.compile(
    r"(\d+\.{1,3})"
    r"|([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)"
    r"|(O-O-O[+#]?|O-O[+#]?)"
    r"|(1-0|0-1|1/2-1/2|\*)"
)

WHITE_MOVE_PREFIX_PATTERN = re.compile(r"^\d+\.")


def tokenize_movetext(text: str) -> list[str]:
    tokens: list[str] = []
    for m in MOVE_TOKEN_PATTERN.finditer(text):
        token = m.group(0)
        if re.match(r"\d+\.{2,}", token):
            token = token.rstrip(".") + "."
        tokens.append(token)
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize SFT JSONL to binary ids")
    parser.add_argument("--input", required=True, help="Input SFT JSONL path")
    parser.add_argument("--vocab", required=True, help="Vocab JSON path")
    parser.add_argument("--output-ids", required=True, help="Output token ids .bin")
    parser.add_argument(
        "--output-loss-mask",
        required=True,
        help="Output loss mask .bin (uint8, 1=loss-on-token)",
    )
    parser.add_argument(
        "--output-stats",
        default="",
        help="Optional output stats JSON path",
    )
    parser.add_argument(
        "--dtype",
        choices=["uint16", "uint32"],
        default="uint16",
        help="Output ids dtype",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--label-eos",
        action="store_true",
        help="Include <eos> token in loss mask",
    )
    parser.add_argument(
        "--map-stm-to-result",
        action="store_true",
        help="Map missing <STM_*> to <white_win>/<draw>/<black_win>",
    )
    return parser.parse_args()


def normalize_completion(completion: str, ply: Optional[int]) -> str:
    completion = completion.strip()
    if not completion:
        return completion
    if WHITE_MOVE_PREFIX_PATTERN.match(completion):
        return completion
    if ply is None or ply % 2 == 0:
        return completion
    move_no = (ply + 1) // 2
    return f"{move_no}.{completion}"


def load_vocab(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    token_to_id = data.get("token_to_id", {})
    if not isinstance(token_to_id, dict):
        raise ValueError("Invalid vocab JSON: token_to_id missing")
    return token_to_id


def maybe_map_control(
    token: str, token_to_id: dict[str, int], map_stm_to_result: bool
) -> str:
    mapping = {
        "<STM_WIN>": "<white_win>",
        "<STM_DRAW>": "<draw>",
        "<STM_LOSS>": "<black_win>",
    }
    if map_stm_to_result:
        return mapping.get(token, token)
    if token in token_to_id:
        return token
    return token


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    vocab_path = Path(args.vocab)
    ids_path = Path(args.output_ids)
    mask_path = Path(args.output_loss_mask)
    stats_path = Path(args.output_stats) if args.output_stats else None

    token_to_id = load_vocab(vocab_path)

    if (
        "<bos>" not in token_to_id
        or "<eos>" not in token_to_id
        or "<unk>" not in token_to_id
    ):
        raise ValueError("Vocab must contain <bos>, <eos>, <unk>")

    unk_id = token_to_id["<unk>"]
    bos_id = token_to_id["<bos>"]
    eos_id = token_to_id["<eos>"]

    np_dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    if args.dtype == "uint16" and len(token_to_id) > 65535:
        raise ValueError("Vocab too large for uint16; use --dtype uint32")

    ids_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    written_samples = 0
    skipped_samples = 0
    total_tokens = 0
    total_unk = 0

    with (
        input_path.open("r", encoding="utf-8", errors="replace") as in_f,
        ids_path.open("wb") as ids_f,
        mask_path.open("wb") as mask_f,
    ):
        for raw_line in in_f:
            line = raw_line.strip()
            if not line:
                continue
            total_samples += 1
            if args.max_samples > 0 and written_samples >= args.max_samples:
                break

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped_samples += 1
                continue

            prompt = str(row.get("prompt", "")).strip()
            completion = str(row.get("completion", "")).strip()
            ply_raw = row.get("ply")
            ply = int(ply_raw) if isinstance(ply_raw, int) else None

            if not prompt or not completion:
                skipped_samples += 1
                continue

            parts = prompt.split(" ", 1)
            control = parts[0]
            movetext_prefix = parts[1] if len(parts) == 2 else ""

            control = maybe_map_control(control, token_to_id, args.map_stm_to_result)
            if control not in token_to_id:
                raise ValueError(
                    f"Control token '{control}' not in vocab. "
                    "Add STM tokens with extend_vocab_for_sft.py or use --map-stm-to-result."
                )

            prompt_tokens = [control] + tokenize_movetext(movetext_prefix)
            completion_text = normalize_completion(completion, ply)
            completion_tokens = tokenize_movetext(completion_text)

            if not completion_tokens:
                skipped_samples += 1
                continue

            sequence_tokens = ["<bos>"] + prompt_tokens + completion_tokens + ["<eos>"]
            ids = [token_to_id.get(tok, unk_id) for tok in sequence_tokens]
            total_unk += sum(1 for i in ids if i == unk_id)

            prompt_len = 1 + len(prompt_tokens)
            completion_len = len(completion_tokens)
            eos_label = 1 if args.label_eos else 0
            mask = [0] * prompt_len + [1] * completion_len + [eos_label]

            np.asarray(ids, dtype=np_dtype).tofile(ids_f)
            np.asarray(mask, dtype=np.uint8).tofile(mask_f)

            total_tokens += len(ids)
            written_samples += 1

            if written_samples % 100000 == 0:
                print(
                    f"written={written_samples} skipped={skipped_samples} "
                    f"tokens={total_tokens}"
                )

    stats = {
        "samples_seen": total_samples,
        "samples_written": written_samples,
        "samples_skipped": skipped_samples,
        "tokens_written": total_tokens,
        "unk_tokens": total_unk,
        "unk_pct": (100.0 * total_unk / total_tokens) if total_tokens else 0.0,
        "dtype": args.dtype,
        "vocab_size": len(token_to_id),
        "label_eos": args.label_eos,
    }

    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    print(
        f"Done. written={written_samples} skipped={skipped_samples} "
        f"tokens={total_tokens} unk={total_unk} ({stats['unk_pct']:.4f}%)"
    )


if __name__ == "__main__":
    main()
