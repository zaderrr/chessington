#!/usr/bin/env python3
"""SFT training on tokenized chess data using ids+lossmask binaries.

Reuses model-loading utilities from ~/Documents/Projects/ai/models/training.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from tokenizers import Tokenizer
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_bool(text: str, default: bool = False) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train chess SFT from tokenized binaries")
    p.add_argument("--base-ckpt", type=Path, required=True)
    p.add_argument("--ids-bin", type=Path, required=True)
    p.add_argument("--loss-mask-bin", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument(
        "--ai-training-dir",
        type=Path,
        default=Path("~/Documents/Projects/ai/models/training").expanduser(),
        help="Directory containing model_loader.py, train_small_llm.py, train_llm.py",
    )

    p.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    p.add_argument("--block-size", type=int, default=0)
    p.add_argument("--train-ratio", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    p.add_argument("--torch-compile", type=str, default="false")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=40)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-warmup-iters", type=int, default=100)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--adam-beta2", type=float, default=0.95)

    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--sample-prompt", type=str, default="<STM_WIN> 1.e4 e5 2.Nf3")
    p.add_argument("--sample-max-new-tokens", type=int, default=60)
    return p.parse_args()


def load_ckpt(path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def lr_for_step(
    step: int, base: float, min_lr: float, warmup: int, total: int
) -> float:
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    if total <= warmup:
        return base
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base - min_lr) * cosine


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def strip_key(key: str) -> str:
        out = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if out.startswith(prefix):
                    out = out[len(prefix) :]
                    changed = True
        return out

    return {strip_key(k): v for k, v in state.items()}


def get_batch(
    ids: np.memmap,
    mask: np.memmap,
    batch_size: int,
    block_size: int,
    split_start: int,
    split_end: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    span = split_end - split_start
    max_start = span - block_size - 1
    if max_start <= 0:
        raise RuntimeError("Split too small for configured block_size")

    starts = np.random.randint(0, max_start, size=(batch_size,), dtype=np.int64)
    starts = starts + split_start

    x_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    for s in starts.tolist():
        x_np = np.asarray(ids[s : s + block_size], dtype=np.int64)
        y_np = np.asarray(ids[s + 1 : s + block_size + 1], dtype=np.int64)
        m_np = np.asarray(mask[s + 1 : s + block_size + 1], dtype=np.uint8)
        y_np[m_np == 0] = -100
        x_rows.append(torch.from_numpy(x_np))
        y_rows.append(torch.from_numpy(y_np))

    xb = torch.stack(x_rows).to(device)
    yb = torch.stack(y_rows).to(device)
    return xb, yb


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_enabled = world_size > 1

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    if not args.base_ckpt.exists():
        raise SystemExit(f"Base checkpoint not found: {args.base_ckpt}")
    if not args.ids_bin.exists():
        raise SystemExit(f"ids bin not found: {args.ids_bin}")
    if not args.loss_mask_bin.exists():
        raise SystemExit(f"loss mask bin not found: {args.loss_mask_bin}")
    if not args.ai_training_dir.exists():
        raise SystemExit(f"ai training dir not found: {args.ai_training_dir}")

    ai_training_dir = str(args.ai_training_dir.resolve())
    if ai_training_dir not in sys.path:
        sys.path.insert(0, ai_training_dir)

    from model_loader import build_model_from_ckpt, detect_model_arch

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    device_text = str(args.device).strip().lower()
    precision = str(args.precision).strip().lower()
    if device_text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        if ddp_enabled:
            args.device = f"cuda:{local_rank}"
        elif device_text == "cuda":
            args.device = "cuda:0"
        torch.cuda.set_device(args.device)
        if precision == "auto":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            precision = "fp32"
    else:
        if precision in {"auto", "bf16"}:
            precision = "fp32"

    if ddp_enabled:
        if not str(args.device).lower().startswith("cuda"):
            raise SystemExit("DDP requires CUDA")
        dist.init_process_group(backend="nccl", init_method="env://")

    ckpt = load_ckpt(args.base_ckpt, args.device)
    cfg = ckpt.get("config") or {}
    meta = ckpt.get("meta") or {}
    ckpt_arch = detect_model_arch(ckpt)
    tokenizer_json = ckpt.get("tokenizer_json")
    if not tokenizer_json:
        raise SystemExit("Checkpoint missing tokenizer_json")
    tokenizer = Tokenizer.from_str(tokenizer_json)
    vocab_size = tokenizer.get_vocab_size()

    ids_dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    ids = np.memmap(args.ids_bin, dtype=ids_dtype, mode="r")
    loss_mask = np.memmap(args.loss_mask_bin, dtype=np.uint8, mode="r")
    if len(ids) != len(loss_mask):
        raise SystemExit("ids and loss mask length mismatch")

    max_token_id = int(np.max(ids)) if len(ids) > 0 else 0
    if max_token_id >= int(vocab_size):
        raise SystemExit(
            "Token id out of range for checkpoint vocab: "
            f"max_id={max_token_id} vocab_size={vocab_size}. "
            "Retokenize with the same vocab as the checkpoint, or use "
            "tokenize_sft_jsonl.py --map-stm-to-result to avoid adding new STM token ids."
        )

    if len(ids) < 1024:
        raise SystemExit("Dataset too small")

    block_size = (
        int(args.block_size)
        if int(args.block_size) > 0
        else int(cfg.get("block_size", 256))
    )
    if block_size < 8:
        raise SystemExit("block_size too small")

    split_idx = int(len(ids) * float(args.train_ratio))
    split_idx = max(block_size + 2, min(split_idx, len(ids) - (block_size + 2)))
    train_start, train_end = 0, split_idx
    val_start, val_end = split_idx, len(ids)

    raw_model, _ = build_model_from_ckpt(
        ckpt,
        vocab_size=vocab_size,
        device=args.device,
        gradient_checkpointing=False,
    )
    if parse_bool(args.torch_compile, default=False) and hasattr(torch, "compile"):
        raw_model = torch.compile(raw_model)
    model = raw_model
    if ddp_enabled:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(args.device).lower().startswith("cuda") and precision == "bf16"
        else nullcontext()
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(0.9, float(args.adam_beta2)),
        weight_decay=float(args.weight_decay),
    )

    eos_id = int(meta.get("eos_id", 0))

    @torch.no_grad()
    def estimate(split: str) -> float:
        model.eval()
        losses: list[float] = []
        s0, s1 = (train_start, train_end) if split == "train" else (val_start, val_end)
        for _ in range(max(1, int(args.eval_iters))):
            xb, yb = get_batch(
                ids,
                loss_mask,
                max(1, int(args.batch_size)),
                block_size,
                s0,
                s1,
                args.device,
            )
            with amp_ctx:
                logits, _ = model(xb, None)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    yb.view(-1),
                    ignore_index=-100,
                )
            losses.append(float(loss.item()))
        model.train()
        mean_loss = sum(losses) / max(1, len(losses))
        if ddp_enabled:
            t = torch.tensor(mean_loss, dtype=torch.float32, device=args.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t = t / max(1, int(world_size))
            mean_loss = float(t.item())
        return mean_loss

    @torch.no_grad()
    def sample_text(prompt: str) -> str:
        src = str(prompt or "").strip()
        if not src:
            return ""
        ids_in = tokenizer.encode(src).ids
        if not ids_in:
            ids_in = [eos_id]
        x = torch.tensor([ids_in], dtype=torch.long, device=args.device)
        out = raw_model.generate(
            x, max_new_tokens=max(1, int(args.sample_max_new_tokens))
        )[0].tolist()
        text = tokenizer.decode(out, skip_special_tokens=True)
        if text.startswith(src):
            text = text[len(src) :].lstrip()
        return text

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start_ts = time.time()

    log(
        f"[dataset] tokens={len(ids)} train={train_end - train_start} val={val_end - val_start} "
        f"block={block_size}"
    )

    model.train()
    for step in range(int(args.max_iters) + 1):
        cur_lr = lr_for_step(
            step,
            float(args.lr),
            float(args.lr_min),
            int(args.lr_warmup_iters),
            int(args.max_iters),
        )
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        if step % max(1, int(args.eval_interval)) == 0:
            train_loss = estimate("train")
            val_loss = estimate("val")
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                if rank == 0:
                    out_ckpt = {
                        "model_state": clean_state_dict(raw_model.state_dict()),
                        "config": cfg,
                        "meta": meta,
                        "tokenizer_json": tokenizer.to_str(),
                        "model_arch": ckpt_arch,
                        "best_val_loss": best_val,
                        "sft": {
                            "ids_bin": str(args.ids_bin),
                            "loss_mask_bin": str(args.loss_mask_bin),
                            "dtype": args.dtype,
                            "train_ratio": float(args.train_ratio),
                            "block_size": int(block_size),
                            "precision": str(precision),
                        },
                    }
                    torch.save(out_ckpt, args.out_dir / "best.pt")

            log(
                f"step {step:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | lr {cur_lr:.6g}"
            )
            if rank == 0 and args.sample_prompt:
                sample = sample_text(args.sample_prompt)
                log(f"[sample] prompt={args.sample_prompt!r} output={sample!r}")

        opt.zero_grad(set_to_none=True)
        for micro_step in range(max(1, int(args.grad_accum_steps))):
            xb, yb = get_batch(
                ids,
                loss_mask,
                max(1, int(args.batch_size)),
                block_size,
                train_start,
                train_end,
                args.device,
            )
            sync_ctx = (
                model.no_sync()  # type: ignore[attr-defined]
                if ddp_enabled and micro_step < (max(1, int(args.grad_accum_steps)) - 1)
                else nullcontext()
            )
            with sync_ctx:
                with amp_ctx:
                    logits, _ = model(xb, None)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        yb.view(-1),
                        ignore_index=-100,
                    )
                (loss / max(1, int(args.grad_accum_steps))).backward()

        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), float(args.grad_clip)
            )
        opt.step()

        if step > 0 and step % max(1, int(args.progress_every)) == 0:
            elapsed_s = max(1e-9, time.time() - start_ts)
            rate = step / elapsed_s
            remaining = max(0, int(args.max_iters) - step)
            eta_s = remaining / max(rate, 1e-9)
            log(
                f"[train] step={step}/{args.max_iters} | rate={rate:.2f} it/s | "
                f"elapsed={elapsed_s:.1f}s | eta={eta_s:.1f}s"
            )

    if rank == 0:
        torch.save(raw_model.state_dict(), args.out_dir / "last_model_state.pt")
        best = args.out_dir / "best.pt"
        if not best.exists():
            out_ckpt = {
                "model_state": clean_state_dict(raw_model.state_dict()),
                "config": cfg,
                "meta": meta,
                "tokenizer_json": tokenizer.to_str(),
                "model_arch": ckpt_arch,
            }
            torch.save(out_ckpt, best)
        log(f"Wrote: {best}")

    if ddp_enabled and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
