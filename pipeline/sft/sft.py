#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer

PRETRAINING_DIR = Path(__file__).resolve().parents[1] / "pre-training"
if str(PRETRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAINING_DIR))

from model_loader import build_model_from_ckpt, detect_model_arch


def read_flow_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "flow_version.txt"
    try:
        text = version_file.read_text(encoding="utf-8").strip()
        return text or "dev"
    except Exception:
        return "dev"


@dataclass
class DatasetSpec:
    dataset: str
    config: str
    split: str
    max_samples: int
    weight: float
    instruction_field: str
    input_field: str
    output_field: str


def parse_bool(value: str, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT fine-tune a chess model checkpoint")
    p.add_argument("--base-ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    # New multi-dataset path.
    p.add_argument("--datasets-json", type=str, default="[]")

    # Backward compatibility with previous single-dataset path.
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=200000)

    p.add_argument("--prompt-template", type=str, default="alpaca")
    p.add_argument("--mask-prompt", type=str, default="true")
    p.add_argument("--instruction-field", type=str, default="instruction")
    p.add_argument("--input-field", type=str, default="input")
    p.add_argument("--output-field", type=str, default="output")

    p.add_argument("--train-ratio", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--precision", type=str, default="auto")
    p.add_argument("--torch-compile", type=str, default="false")
    p.add_argument("--batch-size", type=int, default=4)
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
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--save-best-only", type=str, default="true")
    p.add_argument("--sample-prompts-json", type=str, default="[]")
    p.add_argument("--sample-max-new-tokens", type=int, default=120)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--flow-version", type=str, default="")
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


def normalize_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    rows: list[DatasetSpec] = []
    try:
        parsed = json.loads(str(args.datasets_json or "[]"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --datasets-json: {exc}")
    if isinstance(parsed, list):
        for row in parsed:
            if not isinstance(row, dict):
                continue
            dataset = str(row.get("dataset", "")).strip()
            if not dataset:
                continue
            config = str(row.get("config", "")).strip()
            split = str(row.get("split", "train")).strip() or "train"
            try:
                max_samples = max(0, int(row.get("max_samples", 0)))
            except Exception:
                max_samples = 0
            try:
                weight = float(row.get("weight", 1.0))
            except Exception:
                weight = 1.0
            if weight <= 0:
                continue
            rows.append(
                DatasetSpec(
                    dataset=dataset,
                    config=config,
                    split=split,
                    max_samples=max_samples,
                    weight=weight,
                    instruction_field=str(
                        row.get("instruction_field", args.instruction_field)
                    ).strip()
                    or str(args.instruction_field),
                    input_field=str(row.get("input_field", args.input_field)).strip()
                    or str(args.input_field),
                    output_field=str(row.get("output_field", args.output_field)).strip()
                    or str(args.output_field),
                )
            )

    if rows:
        return rows

    legacy_dataset = str(args.dataset or "").strip()
    if legacy_dataset:
        return [
            DatasetSpec(
                dataset=legacy_dataset,
                config=str(args.config or "").strip(),
                split=str(args.split or "train").strip() or "train",
                max_samples=max(0, int(args.max_samples or 0)),
                weight=1.0,
                instruction_field=str(args.instruction_field),
                input_field=str(args.input_field),
                output_field=str(args.output_field),
            )
        ]
    raise SystemExit("At least one dataset is required (datasets-json or --dataset)")


def format_prompt(
    template: str,
    instruction: str,
    input_text: str,
    output_text: str,
) -> tuple[str, str]:
    t = str(template or "alpaca").strip().lower()
    if t == "chatml":
        user_text = instruction
        if input_text:
            user_text = f"{instruction}\n\n{input_text}" if instruction else input_text
        prompt = f"<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"
        response = f"{output_text}\n<|im_end|>"
        return prompt, response
    if t == "vicuna":
        user_text = instruction
        if input_text:
            user_text = f"{instruction}\n{input_text}" if instruction else input_text
        prompt = f"USER: {user_text}\nASSISTANT: "
        response = output_text
        return prompt, response
    # alpaca and custom both map to this stable default.
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"\n### Input:\n{input_text}\n"
    prompt += "\n### Response:\n"
    response = output_text
    return prompt, response


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_enabled = world_size > 1

    def log(msg: str, *, flush: bool = True) -> None:
        if rank == 0:
            print(msg, flush=flush)

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

    script_flow_version = read_flow_version()
    expected_flow_version = str(args.flow_version or "").strip()
    if expected_flow_version and expected_flow_version != script_flow_version:
        raise SystemExit(
            "Flow version mismatch for SFT script: "
            f"expected={expected_flow_version} actual={script_flow_version}"
        )
    log(
        f"[flow] stage=sft script_version={script_flow_version}",
        flush=True,
    )
    if not args.base_ckpt.exists():
        raise SystemExit(f"Base checkpoint not found: {args.base_ckpt}")
    if int(args.batch_size) <= 0 or int(args.grad_accum_steps) <= 0:
        raise SystemExit("batch_size and grad_accum_steps must be >= 1")
    if float(args.lr_min) < 0:
        raise SystemExit("lr_min must be >= 0")
    if float(args.weight_decay) < 0:
        raise SystemExit("weight_decay must be >= 0")
    if float(args.adam_beta2) <= 0 or float(args.adam_beta2) >= 1:
        raise SystemExit("adam_beta2 must be in (0, 1)")

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "datasets package is required for SFT. Install: pip install datasets"
        ) from exc

    datasets_cfg = normalize_datasets(args)
    mask_prompt = parse_bool(args.mask_prompt, default=True)
    use_torch_compile = parse_bool(args.torch_compile, default=False)
    save_best_only = parse_bool(args.save_best_only, default=True)
    try:
        sample_prompts_raw = json.loads(str(args.sample_prompts_json or "[]"))
    except json.JSONDecodeError:
        sample_prompts_raw = []
    sample_prompts: list[dict[str, str]] = []
    if isinstance(sample_prompts_raw, list):
        for row in sample_prompts_raw:
            if isinstance(row, dict):
                prompt = str(row.get("prompt", "")).strip()
                category = str(row.get("category", "")).strip()
            else:
                prompt = str(row).strip()
                category = ""
            if not prompt:
                continue
            sample_prompts.append({"prompt": prompt, "category": category})
    sample_prompts = sample_prompts[:12]

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    device_text = str(args.device).strip().lower()
    precision = str(args.precision or "auto").strip().lower() or "auto"
    if precision not in {"auto", "fp32", "bf16"}:
        raise SystemExit("precision must be one of: auto, fp32, bf16")
    if device_text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        if ddp_enabled:
            args.device = f"cuda:{local_rank}"
        elif str(args.device).strip().lower() == "cuda":
            args.device = "cuda:0"
        torch.cuda.set_device(args.device)
        bf16_ok = bool(torch.cuda.is_bf16_supported())
        if precision == "auto":
            precision = "bf16" if bf16_ok else "fp32"
        elif precision == "bf16" and not bf16_ok:
            precision = "fp32"
    elif device_text == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available")
        if precision in {"auto", "bf16"}:
            precision = "fp32"
    else:
        if precision in {"auto", "bf16"}:
            precision = "fp32"

    if ddp_enabled:
        if not str(args.device).lower().startswith("cuda"):
            raise SystemExit("DDP mode requires CUDA")
        dist.init_process_group(backend="nccl", init_method="env://")
        log(f"[ddp] enabled world_size={world_size}", flush=True)

    ckpt = load_ckpt(args.base_ckpt, args.device)
    cfg = ckpt.get("config") or {}
    meta = ckpt.get("meta") or {}
    ckpt_arch = detect_model_arch(ckpt)
    tokenizer_json = ckpt.get("tokenizer_json")
    if not tokenizer_json:
        raise SystemExit("Checkpoint missing tokenizer_json")
    tokenizer = Tokenizer.from_str(tokenizer_json)
    vocab_size = tokenizer.get_vocab_size()

    raw_model, _ = build_model_from_ckpt(
        ckpt,
        vocab_size=vocab_size,
        device=args.device,
        gradient_checkpointing=False,
    )
    if use_torch_compile and hasattr(torch, "compile"):
        raw_model = torch.compile(raw_model)
    model = raw_model
    if ddp_enabled:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)

    block_size = int(cfg.get("block_size", 256))
    eos_id = int(meta.get("eos_id", 0))
    do_lower = bool(meta.get("lowercase", False))

    train_rows_by_ds: list[list[dict]] = []
    val_rows_by_ds: list[list[dict]] = []
    train_weights: list[float] = []
    train_total = 0
    val_total = 0

    for idx, spec in enumerate(datasets_cfg, start=1):
        kwargs = {"path": spec.dataset, "split": spec.split}
        if spec.config:
            kwargs["name"] = spec.config
        hf_token = str(
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HF_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or ""
        ).strip()
        if hf_token:
            kwargs["token"] = hf_token
        log(
            f"[dataset] loading {idx}/{len(datasets_cfg)} id={spec.dataset} split={spec.split} config={spec.config or '-'}",
            flush=True,
        )
        ds = load_dataset(**kwargs).shuffle(seed=int(args.seed) + idx)
        if spec.max_samples > 0:
            ds = ds.select(range(min(len(ds), int(spec.max_samples))))
        if len(ds) < 2:
            continue
        split = ds.train_test_split(
            train_size=float(args.train_ratio), seed=int(args.seed)
        )
        train_rows = list(split["train"])
        val_rows = list(split["test"])
        if not train_rows or not val_rows:
            continue
        train_rows_by_ds.append(train_rows)
        val_rows_by_ds.append(val_rows)
        train_weights.append(float(spec.weight))
        train_total += len(train_rows)
        val_total += len(val_rows)
        log(
            f"[dataset] ready id={spec.dataset} train={len(train_rows)} val={len(val_rows)} weight={spec.weight}",
            flush=True,
        )

    if not train_rows_by_ds or not val_rows_by_ds:
        raise SystemExit("No usable train/validation rows after dataset loading")

    log(
        f"[dataset] combined train={train_total} val={val_total} datasets={len(train_rows_by_ds)}",
        flush=True,
    )

    def row_to_text(row: dict, spec: DatasetSpec) -> tuple[str, str] | None:
        instruction_key = str(
            spec.instruction_field or args.instruction_field
        ).strip() or str(args.instruction_field)
        input_key = str(spec.input_field or args.input_field).strip() or str(
            args.input_field
        )
        output_key = str(spec.output_field or args.output_field).strip() or str(
            args.output_field
        )
        instruction = str(row.get(instruction_key, "")).strip()
        input_text = str(row.get(input_key, "")).strip()
        output_text = str(row.get(output_key, "")).strip()
        if not output_text:
            return None
        prompt, response = format_prompt(
            template=str(args.prompt_template),
            instruction=instruction,
            input_text=input_text,
            output_text=output_text,
        )
        if do_lower:
            return prompt.lower(), response.lower()
        return prompt, response

    @torch.no_grad()
    def generate_text(prompt_text: str) -> str:
        src = str(prompt_text or "").strip()
        if not src:
            return ""
        if do_lower:
            src = src.lower()
        ids = tokenizer.encode(src).ids
        if not ids:
            ids = [eos_id]
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        out = raw_model.generate(
            x, max_new_tokens=max(1, int(args.sample_max_new_tokens))
        )[0].tolist()
        text = tokenizer.decode(out, skip_special_tokens=True)
        # keep only generation suffix when prompt is present as prefix
        if text.startswith(src):
            text = text[len(src) :].lstrip()
        return text

    def sample_batch(
        rows_by_ds: list[list[dict]],
        specs_by_ds: list[DatasetSpec],
        weights: list[float],
        batch_size: int,
    ):
        xs: list[list[int]] = []
        ys: list[list[int]] = []
        while len(xs) < batch_size:
            ds_idx = random.choices(range(len(rows_by_ds)), weights=weights, k=1)[0]
            rows = rows_by_ds[ds_idx]
            spec = specs_by_ds[ds_idx]
            row = rows[random.randrange(len(rows))]
            text_pair = row_to_text(row, spec)
            if not text_pair:
                continue
            prompt_text, response_text = text_pair
            prompt_ids = tokenizer.encode(prompt_text).ids
            response_ids = tokenizer.encode(response_text).ids
            if not response_ids:
                continue
            full_ids = (prompt_ids + response_ids)[: block_size - 1]
            full_ids.append(eos_id)
            if len(full_ids) < 2:
                continue

            x = full_ids[:-1]
            y = full_ids[1:]

            if mask_prompt:
                prompt_len = min(len(prompt_ids), len(full_ids))
                ignore_until = max(0, min(len(y), prompt_len - 1))
                for i in range(ignore_until):
                    y[i] = -100

            if len(x) < block_size:
                x_pad = [eos_id] * (block_size - len(x))
                y_pad = [-100] * (block_size - len(y))
                x = x + x_pad
                y = y + y_pad

            xs.append(x)
            ys.append(y)

        xb = torch.tensor(xs, dtype=torch.long, device=args.device)
        yb = torch.tensor(ys, dtype=torch.long, device=args.device)
        return xb, yb

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

    @torch.no_grad()
    def estimate(
        rows_by_ds: list[list[dict]],
        specs_by_ds: list[DatasetSpec],
        weights: list[float],
    ) -> float:
        model.eval()
        losses: list[float] = []
        for _ in range(max(1, int(args.eval_iters))):
            xb, yb = sample_batch(
                rows_by_ds, specs_by_ds, weights, max(1, int(args.batch_size))
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
        split_mean = sum(losses) / max(1, len(losses))
        if ddp_enabled:
            t = torch.tensor(split_mean, dtype=torch.float32, device=args.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t = t / max(1, int(world_size))
            split_mean = float(t.item())
        return split_mean

    best_val = float("inf")
    no_improve_evals = 0
    early_patience = max(0, int(args.early_stopping_patience))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_ts = time.time()

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
            train_loss = estimate(train_rows_by_ds, datasets_cfg, train_weights)
            val_loss = estimate(val_rows_by_ds, datasets_cfg, train_weights)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                no_improve_evals = 0
                if rank == 0:
                    out_ckpt = {
                        "model_state": clean_state_dict(raw_model.state_dict()),
                        "config": cfg,
                        "meta": meta,
                        "flow_version": script_flow_version,
                        "tokenizer_json": tokenizer.to_str(),
                        "model_arch": ckpt_arch,
                        "best_val_loss": best_val,
                        "sft": {
                            "datasets": [x.__dict__ for x in datasets_cfg],
                            "prompt_template": str(args.prompt_template),
                            "mask_prompt": bool(mask_prompt),
                            "instruction_field": str(args.instruction_field),
                            "input_field": str(args.input_field),
                            "output_field": str(args.output_field),
                            "train_ratio": float(args.train_ratio),
                            "precision": str(precision),
                            "ddp": bool(ddp_enabled),
                            "world_size": int(world_size),
                        },
                    }
                    torch.save(out_ckpt, args.out_dir / "best.pt")
            else:
                no_improve_evals += 1

            log(
                f"step {step:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | lr {cur_lr:.6g}",
                flush=True,
            )
            if improved:
                log(f"[eval] new_best val={best_val:.4f}", flush=True)
            if early_patience > 0 and no_improve_evals >= early_patience:
                log(
                    f"[early_stop] no val improvement for {no_improve_evals} evals (patience={early_patience})",
                    flush=True,
                )
                break

            if sample_prompts and rank == 0:
                for row in sample_prompts:
                    prompt = str(row.get("prompt", "")).strip()
                    category = str(row.get("category", "")).strip()
                    if not prompt:
                        continue
                    sample = generate_text(prompt)
                    log(
                        "[sample] "
                        + json.dumps(
                            {
                                "step": int(step),
                                "category": category,
                                "prompt": prompt,
                                "output": sample,
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )

        opt.zero_grad(set_to_none=True)
        for micro_step in range(max(1, int(args.grad_accum_steps))):
            xb, yb = sample_batch(
                train_rows_by_ds,
                datasets_cfg,
                train_weights,
                max(1, int(args.batch_size)),
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

        grad_clip = max(0.0, float(args.grad_clip))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        opt.step()

        if step > 0 and step % max(1, int(args.progress_every)) == 0:
            elapsed_s = max(1e-9, time.time() - start_ts)
            rate = step / elapsed_s
            remaining = max(0, int(args.max_iters) - step)
            eta_s = remaining / max(rate, 1e-9)
            log(
                f"[train] step={step}/{args.max_iters} | rate={rate:.2f} it/s | elapsed={elapsed_s:.1f}s | eta={eta_s:.1f}s | lr={cur_lr:.6g} | grad_accum={args.grad_accum_steps}",
                flush=True,
            )

    if rank == 0:
        if not save_best_only:
            torch.save(raw_model.state_dict(), args.out_dir / "last_model_state.pt")
        else:
            # Keep compatibility with existing artifact expectations.
            torch.save(raw_model.state_dict(), args.out_dir / "last_model_state.pt")

    best = args.out_dir / "best.pt"
    if rank == 0 and not best.exists():
        out_ckpt = {
            "model_state": clean_state_dict(raw_model.state_dict()),
            "config": cfg,
            "meta": meta,
            "flow_version": script_flow_version,
            "tokenizer_json": tokenizer.to_str(),
            "model_arch": ckpt_arch,
            "sft": {"datasets": [x.__dict__ for x in datasets_cfg]},
        }
        torch.save(out_ckpt, best)
    log(f"Wrote: {best}", flush=True)

    if ddp_enabled and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
