#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F

TOKENISATION_DIR = Path(__file__).resolve().parents[1] / "tokenisation"
if str(TOKENISATION_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENISATION_DIR))

PRETRAINING_DIR = Path(__file__).resolve().parents[1] / "pre-training"
if str(PRETRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_DIR))

from tokenizer_utils import sanitize_text
from model_loader import build_model_from_ckpt


def load_checkpoint(path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def sample(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def build_model_from_checkpoint(
    ckpt: dict, device: str
) -> tuple[nn.Module, Tokenizer, dict, str]:
    if "config" not in ckpt or "meta" not in ckpt:
        raise SystemExit(
            "Checkpoint missing config/meta dict"
        )
    if "tokenizer_json" not in ckpt:
        raise SystemExit(
            "Checkpoint missing tokenizer payload"
        )

    cfg = ckpt["config"]
    meta = ckpt["meta"]
    tokenizer = Tokenizer.from_str(ckpt["tokenizer_json"])
    vocab_size = tokenizer.get_vocab_size()

    model, arch = build_model_from_ckpt(
        ckpt,
        vocab_size=vocab_size,
        device=device,
        gradient_checkpointing=False,
    )
    model.eval()
    return model, tokenizer, meta, arch


def run_once(
    model: nn.Module,
    tokenizer: Tokenizer,
    meta: dict,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> str:
    prompt_text = sanitize_text(prompt)
    if meta.get("lowercase", False):
        prompt_text = prompt_text.lower()
    prompt_ids = tokenizer.encode(prompt_text).ids
    if not prompt_ids:
        prompt_ids = [meta.get("unk_id", 0)]

    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = sample(model, context, max_new_tokens, temperature, top_k)[0].tolist()
    completion_ids = out[len(prompt_ids) :]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    if completion_text:
        return completion_text
    return tokenizer.decode(out, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for a chess model checkpoint")
    p.add_argument(
        "--ckpt", type=Path, default=Path("storage/out/chess/best.pt")
    )
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--prompt", type=str, default="1. e4 e5 2. Nf3 ")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--interactive", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    ckpt = load_checkpoint(args.ckpt, args.device)
    model, tokenizer, meta, arch = build_model_from_checkpoint(ckpt, args.device)
    print(f"[arch] {arch}")

    if args.interactive:
        print("Loaded model. Type prompts and press Enter. Type :q to quit.")
        while True:
            try:
                prompt = input("\nPrompt> ")
            except EOFError:
                break
            if prompt.strip() in {":q", ":quit", "quit", "exit"}:
                break
            output = run_once(
                model,
                tokenizer,
                meta,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.device,
            )
            print("\n--- Output ---")
            print(output)
    else:
        output = run_once(
            model,
            tokenizer,
            meta,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.device,
        )
        print(output)


if __name__ == "__main__":
    main()
