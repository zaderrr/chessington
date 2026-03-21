#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tokenizers import Tokenizer
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

TOKENIZATION_DIR = Path(__file__).resolve().parents[1] / "tokenisation"
if str(TOKENIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZATION_DIR))

from tokenizer_utils import sanitize_text


@dataclass
class TrainConfig:
    data_dirs: list[Path] = field(
        default_factory=lambda: [Path("storage/data/chess")]
    )
    data_weights_json: str = ""
    out_dir: Path = Path("storage/out/chess")
    batch_size: int = 64
    grad_accum_steps: int = 1
    block_size: int = 256
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 100
    eval_batch_size: int = 0
    lr: float = 3e-4
    lr_warmup_iters: int = 0
    lr_min: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    use_fused_adamw: bool = True
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "auto"
    seed: int = 1337
    gen_tokens: int = 300
    gen_prompt: str = "1. e4 e5 2. Nf3 "
    show_progress: bool = True
    tb_logdir: str = ""
    progress_every: int = 0
    checkpoint_interval: int = 0
    resume: Path | None = None
    flow_version: str = ""
    gradient_checkpointing: bool = False
    gen_temperature: float = 0.8
    gen_top_k: int = 50


def read_flow_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "flow_version.txt"
    try:
        text = version_file.read_text(encoding="utf-8").strip()
        return text or "dev"
    except Exception:
        return "dev"


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._max_cached:
            return
        self._max_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]  # (1, 1, seq, dim)
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embd: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.attn_dropout = float(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(head_size, max_seq_len=block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, t, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rotary(t)
        q, k = apply_rotary_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, t, c)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    """SwiGLU feedforward: better performance per parameter than GELU 4x."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        # SwiGLU uses ~8/3 expansion to match param count of 4x GELU
        hidden = int(2 * (4 * n_embd) / 3)
        # Round to nearest multiple of 64 for hardware efficiency
        hidden = 64 * ((hidden + 63) // 64)
        self.gate_proj = nn.Linear(n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTMini(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.block_size = block_size
        self.gradient_checkpointing = gradient_checkpointing
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # RoPE handles position info — no learned position embeddings needed
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
        # Scale residual projections per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, t = idx.shape
        x = self.token_embedding_table(idx)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, tt, c = logits.shape
            loss = F.cross_entropy(logits.view(b * tt, c), targets.view(b * tt))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def chunked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, chunk_size: int = 1024
) -> torch.Tensor:
    """Compute cross entropy in chunks to avoid peak memory spike from large vocab."""
    total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    total_tokens = logits.size(0)

    for i in range(0, total_tokens, chunk_size):
        chunk_logits = logits[i : i + chunk_size]
        chunk_targets = targets[i : i + chunk_size]
        chunk_loss = F.cross_entropy(chunk_logits, chunk_targets, reduction="sum")
        total_loss += chunk_loss

    return total_loss / total_tokens


def load_data(data_dir: Path) -> tuple[np.memmap, np.memmap, dict]:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    dtype = np.dtype(meta["dtype"])
    train_data = np.memmap(data_dir / "train.bin", dtype=dtype, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=dtype, mode="r")
    return train_data, val_data, meta


def _tokenizer_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_multi_data(
    data_dirs: list[Path],
) -> tuple[list[np.memmap], list[np.memmap], dict, Path]:
    if not data_dirs:
        raise SystemExit("At least one data directory is required")

    resolved_dirs = [Path(d).expanduser().resolve() for d in data_dirs]
    train_sets: list[np.memmap] = []
    val_sets: list[np.memmap] = []
    metas: list[dict] = []

    tokenizer_digest: str | None = None
    tokenizer_path: Path | None = None
    dtype0: str | None = None
    vocab0: int | None = None

    for data_dir in resolved_dirs:
        train_data, val_data, meta = load_data(data_dir)
        train_sets.append(train_data)
        val_sets.append(val_data)
        metas.append(meta)

        dtype = str(meta.get("dtype", ""))
        vocab = int(meta.get("vocab_size", 0))
        if dtype0 is None:
            dtype0 = dtype
            vocab0 = vocab
        else:
            if dtype != dtype0:
                raise SystemExit(
                    f"Incompatible dtype across data dirs: expected={dtype0} got={dtype} dir={data_dir}"
                )
            if vocab != vocab0:
                raise SystemExit(
                    f"Incompatible vocab_size across data dirs: expected={vocab0} got={vocab} dir={data_dir}"
                )

        tok_rel = str(meta.get("tokenizer_path", "tokenizer.json") or "tokenizer.json")
        tok_path = (data_dir / tok_rel).resolve()
        if not tok_path.exists():
            raise SystemExit(f"Missing tokenizer file for data dir: {tok_path}")
        digest = _tokenizer_sha256(tok_path)
        if tokenizer_digest is None:
            tokenizer_digest = digest
            tokenizer_path = tok_path
        elif digest != tokenizer_digest:
            raise SystemExit(
                "Incompatible tokenizer.json across data dirs; expected identical tokenizer artifacts "
                f"for mixed training. dir={data_dir}"
            )

    if tokenizer_path is None:
        raise SystemExit("Failed to resolve tokenizer for data directories")

    merged_meta = dict(metas[0])
    merged_meta["num_train_tokens"] = int(
        sum(int(meta.get("num_train_tokens", 0)) for meta in metas)
    )
    merged_meta["num_val_tokens"] = int(
        sum(int(meta.get("num_val_tokens", 0)) for meta in metas)
    )
    merged_meta["merged_from"] = [str(d) for d in resolved_dirs]

    return train_sets, val_sets, merged_meta, tokenizer_path


def parse_data_weights_json(raw: str, expected_len: int) -> list[float] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --data-weights-json: {exc}")
    if not isinstance(parsed, list):
        raise SystemExit("--data-weights-json must be a JSON array of numbers")
    if len(parsed) != int(expected_len):
        raise SystemExit(
            f"--data-weights-json length mismatch: expected {expected_len}, got {len(parsed)}"
        )
    out: list[float] = []
    for idx, value in enumerate(parsed):
        try:
            fv = float(value)
        except (TypeError, ValueError):
            raise SystemExit(f"Invalid data weight at index {idx}: {value!r}")
        if not math.isfinite(fv) or fv <= 0:
            raise SystemExit(f"Data weight must be > 0 at index {idx}: {value!r}")
        out.append(fv)
    return out


def get_batch(
    data: np.memmap | list[np.memmap],
    batch_size: int,
    block_size: int,
    device: torch.device,
    sample_weights: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(data, list):
        return get_batch_multi(
            data,
            batch_size,
            block_size,
            device,
            sample_weights=sample_weights,
        )
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + block_size + 1]).astype(np.int64))
            for i in ix
        ]
    )
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_batch_multi(
    data_sets: list[np.memmap],
    batch_size: int,
    block_size: int,
    device: torch.device,
    sample_weights: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not data_sets:
        raise RuntimeError("No datasets available for batch sampling")

    max_starts = [len(d) - block_size - 1 for d in data_sets]
    valid = [i for i, m in enumerate(max_starts) if m > 0]
    if not valid:
        raise RuntimeError(
            "All selected datasets are too small for the configured block_size"
        )

    if sample_weights is not None:
        if len(sample_weights) != len(data_sets):
            raise RuntimeError(
                f"sample_weights size mismatch: expected {len(data_sets)} got {len(sample_weights)}"
            )
        weights = torch.tensor(
            [max(0.0, float(sample_weights[i])) for i in valid], dtype=torch.float32
        )
        if float(weights.sum().item()) <= 0:
            raise RuntimeError("All effective dataset weights are zero")
    else:
        weights = torch.tensor([max_starts[i] for i in valid], dtype=torch.float32)
    picks = torch.multinomial(weights, num_samples=batch_size, replacement=True)

    x_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    for pick in picks.tolist():
        data_idx = valid[pick]
        data = data_sets[data_idx]
        max_start = max_starts[data_idx]
        start = int(torch.randint(max_start, (1,)).item())
        x_rows.append(
            torch.from_numpy((data[start : start + block_size]).astype(np.int64))
        )
        y_rows.append(
            torch.from_numpy(
                (data[start + 1 : start + block_size + 1]).astype(np.int64)
            )
        )

    x = torch.stack(x_rows)
    y = torch.stack(y_rows)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: np.memmap | list[np.memmap],
    val_data: np.memmap | list[np.memmap],
    cfg: TrainConfig,
    sample_weights: list[float] | None,
    *,
    ddp_enabled: bool,
    world_size: int,
    rank: int,
):
    out = {}
    model.eval()
    eval_batch = (
        int(cfg.eval_batch_size)
        if int(cfg.eval_batch_size) > 0
        else int(cfg.batch_size)
    )
    eval_log_every = max(1, int(cfg.eval_iters) // 5)
    for split, data in (("train", train_data), ("val", val_data)):
        if rank == 0:
            print(
                f"[eval] split={split} step_iters={cfg.eval_iters} batch={eval_batch} block={cfg.block_size}",
                flush=True,
            )
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(
                data,
                eval_batch,
                cfg.block_size,
                torch.device(cfg.device),
                sample_weights=sample_weights,
            )
            amp_ctx = _autocast_ctx(cfg)
            with amp_ctx:
                _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Model returned no loss during evaluation")
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            losses[k] = float(loss.item())
            if rank == 0 and (
                (k + 1) % eval_log_every == 0 or (k + 1) == int(cfg.eval_iters)
            ):
                print(
                    f"[eval] split={split} progress={k + 1}/{cfg.eval_iters}",
                    flush=True,
                )
        split_mean = losses.mean().to(torch.device(cfg.device))
        if ddp_enabled:
            dist.all_reduce(split_mean, op=dist.ReduceOp.SUM)
            split_mean = split_mean / max(1, int(world_size))
        out[split] = float(split_mean.item())
    model.train()
    return out


def _capture_rng_state(device_text: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state().cpu(),
        "numpy": np.random.get_state(),
    }
    if device_text.startswith("cuda") and torch.cuda.is_available():
        try:
            state["cuda"] = [x.cpu() for x in torch.cuda.get_rng_state_all()]
        except Exception:
            pass
    return state


def _restore_rng_state(
    state: dict[str, Any] | None,
    *,
    device_text: str,
    ddp_enabled: bool,
    rank: int,
    log: Callable[[str], None],
) -> None:
    if not isinstance(state, dict):
        return
    if ddp_enabled:
        if rank == 0:
            log("[resume] skipping RNG restore in DDP mode")
        return
    try:
        torch_state = state.get("torch")
        if isinstance(torch_state, torch.Tensor):
            torch.set_rng_state(torch_state.cpu())
    except Exception as exc:
        log(f"[resume] failed to restore torch RNG: {exc}")
    try:
        np_state = state.get("numpy")
        if np_state is not None:
            np.random.set_state(np_state)
    except Exception as exc:
        log(f"[resume] failed to restore numpy RNG: {exc}")
    if device_text.startswith("cuda") and torch.cuda.is_available():
        try:
            cuda_state = state.get("cuda")
            if isinstance(cuda_state, list) and cuda_state:
                torch.cuda.set_rng_state_all([x.to("cuda") for x in cuda_state])
        except Exception as exc:
            log(f"[resume] failed to restore CUDA RNG: {exc}")


def _build_checkpoint(
    *,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    meta: dict,
    script_flow_version: str,
    tokenizer: Tokenizer,
    best_val: float,
    step: int,
    ddp_enabled: bool,
    world_size: int,
    device_text: str,
) -> dict[str, Any]:
    return {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": vars(cfg),
        "meta": meta,
        "model_arch": "gptmini_v2",
        "flow_version": script_flow_version,
        "tokenizer_json": tokenizer.to_str(),
        "best_val_loss": float(best_val),
        "best_val": float(best_val),
        "step": int(step),
        "rng_state": _capture_rng_state(device_text),
        "ddp": bool(ddp_enabled),
        "world_size": int(world_size),
        "saved_at": time.time(),
    }


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Pre-train a chess model on tokenised PGN data")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--data-dirs", type=Path, nargs="+", default=None)
    p.add_argument("--data-weights-json", type=str, default="")
    p.add_argument(
        "--out-dir", type=Path, default=Path("storage/out/chess")
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warmup-iters", type=int, default=0)
    p.add_argument("--lr-min", type=float, default=0.0)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-fused-adamw", type=str, default="true")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--precision", type=str, default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--gen-tokens", type=int, default=300)
    p.add_argument("--gen-prompt", type=str, default="1. e4 e5 2. Nf3 ")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--tb-logdir", type=str, default="")
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--checkpoint-interval", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--flow-version", type=str, default="")
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--gen-temperature", type=float, default=0.8)
    p.add_argument("--gen-top-k", type=int, default=50)
    args = p.parse_args()
    arg_dict = vars(args)

    data_dirs = arg_dict.pop("data_dirs", None)
    data_dir = arg_dict.pop("data_dir", None)
    if data_dirs:
        arg_dict["data_dirs"] = list(data_dirs)
    elif data_dir is not None:
        arg_dict["data_dirs"] = [data_dir]
    else:
        arg_dict["data_dirs"] = [Path("storage/data/chess")]

    arg_dict["use_fused_adamw"] = str(
        arg_dict.get("use_fused_adamw", "true")
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    arg_dict["show_progress"] = not arg_dict.pop("no_progress")
    return TrainConfig(**arg_dict)


def main() -> None:
    cfg = parse_args()
    script_flow_version = read_flow_version()
    expected_flow_version = str(cfg.flow_version or "").strip()
    if expected_flow_version and expected_flow_version != script_flow_version:
        raise SystemExit(
            "Flow version mismatch for training script: "
            f"expected={expected_flow_version} actual={script_flow_version}"
        )
    if int(cfg.grad_accum_steps) <= 0:
        raise SystemExit("grad_accum_steps must be >= 1")
    if float(cfg.lr_min) < 0:
        raise SystemExit("lr_min must be >= 0")
    if not (0.0 < float(cfg.adam_beta1) < 1.0 and 0.0 < float(cfg.adam_beta2) < 1.0):
        raise SystemExit("adam_beta1 and adam_beta2 must be in (0, 1)")
    if float(cfg.weight_decay) < 0:
        raise SystemExit("weight_decay must be >= 0")
    if float(cfg.grad_clip) < 0:
        raise SystemExit("grad_clip must be >= 0")
    if int(cfg.checkpoint_interval) < 0:
        raise SystemExit("checkpoint_interval must be >= 0")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_enabled = world_size > 1

    def log(msg: str, *, flush: bool = True) -> None:
        if rank == 0:
            print(msg, flush=flush)

    alloc = (
        str(
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            or os.environ.get("PYTORCH_ALLOC_CONF")
            or ""
        ).strip()
        or "expandable_segments:True"
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", alloc)
    log(
        f"[alloc_conf] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}",
        flush=True,
    )
    log(
        f"[flow] stage=training script_version={script_flow_version}",
        flush=True,
    )

    device_text = str(cfg.device).lower()
    precision_text = str(cfg.precision).strip().lower() or "auto"
    if precision_text not in {"auto", "fp32", "bf16"}:
        raise SystemExit("precision must be one of: auto, fp32, bf16")
    cfg.precision = precision_text
    if device_text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        resolved_cuda = cfg.device
        if ddp_enabled:
            resolved_cuda = f"cuda:{local_rank}"
        elif str(cfg.device).strip().lower() == "cuda":
            resolved_cuda = "cuda:0"
        try:
            torch.cuda.set_device(resolved_cuda)
        except Exception as exc:
            raise SystemExit(f"Failed to select CUDA device '{cfg.device}': {exc}")
        cfg.device = str(resolved_cuda)
        idx = torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(idx)
        bf16_ok = bool(torch.cuda.is_bf16_supported())
        if cfg.precision == "auto":
            cfg.precision = "bf16" if bf16_ok else "fp32"
        elif cfg.precision == "bf16" and not bf16_ok:
            log(
                "[precision] bf16 requested but unsupported; falling back to fp32",
                flush=True,
            )
            cfg.precision = "fp32"
        log(
            f"[device] cuda index={idx} name={torch.cuda.get_device_name(idx)} free={free_b / 1024**3:.2f}GiB total={total_b / 1024**3:.2f}GiB"
        )
        log(f"[precision] {cfg.precision}", flush=True)
    elif device_text == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available")
        if cfg.precision == "auto":
            cfg.precision = "fp32"
        elif cfg.precision == "bf16":
            log(
                "[precision] bf16 not supported on MPS path here; falling back to fp32",
                flush=True,
            )
            cfg.precision = "fp32"
        log("[device] using mps")
        log(f"[precision] {cfg.precision}", flush=True)
    else:
        if cfg.precision == "auto":
            cfg.precision = "fp32"
        elif cfg.precision == "bf16":
            log("[precision] bf16 requested on CPU; falling back to fp32", flush=True)
            cfg.precision = "fp32"
        log("[device] using cpu")
        log(f"[precision] {cfg.precision}", flush=True)

    if ddp_enabled:
        if not str(cfg.device).lower().startswith("cuda"):
            raise SystemExit("DDP mode requires CUDA")
        dist.init_process_group(backend="nccl", init_method="env://")
        log(f"[ddp] enabled world_size={world_size}", flush=True)
    elif str(cfg.device).lower().startswith("cuda"):
        log(f"[multi_gpu] single-process mode visible_gpus={torch.cuda.device_count()}")

    torch.manual_seed(int(cfg.seed) + rank)
    np.random.seed(int(cfg.seed) + rank)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, meta, tokenizer_path = load_multi_data(cfg.data_dirs)
    data_sample_weights = parse_data_weights_json(
        cfg.data_weights_json, expected_len=len(cfg.data_dirs)
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    log(
        "[data] sources="
        + ", ".join(str(Path(d).expanduser().resolve()) for d in cfg.data_dirs)
        + f" | train_tokens={int(meta.get('num_train_tokens', 0))}"
        + f" | val_tokens={int(meta.get('num_val_tokens', 0))}",
        flush=True,
    )
    if data_sample_weights is not None:
        weights_text = ", ".join(f"{w:.6f}" for w in data_sample_weights)
        log(f"[data] sampling_weights=[{weights_text}]", flush=True)

    if cfg.n_embd % cfg.n_head != 0:
        raise SystemExit("n_embd must be divisible by n_head")

    raw_model = GPTMini(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(cfg.device)
    model: nn.Module = raw_model
    if ddp_enabled:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)

    adamw_kwargs = {
        "lr": float(cfg.lr),
        "betas": (float(cfg.adam_beta1), float(cfg.adam_beta2)),
    }

    decay_names: set[str] = set()
    no_decay_names: set[str] = set()
    for module_name, module in raw_model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if param_name.endswith("bias"):
                no_decay_names.add(full_name)
            elif isinstance(module, (nn.LayerNorm, nn.Embedding, RMSNorm)):
                no_decay_names.add(full_name)
            else:
                decay_names.add(full_name)

    param_dict = {
        name: p for name, p in raw_model.named_parameters() if p.requires_grad
    }
    decay_names &= set(param_dict.keys())
    no_decay_names &= set(param_dict.keys())
    inter = decay_names & no_decay_names
    if inter:
        raise SystemExit(f"Parameter grouping overlap detected: {sorted(inter)[:5]}")
    missing = set(param_dict.keys()) - (decay_names | no_decay_names)
    if missing:
        no_decay_names |= missing

    optim_groups = [
        {
            "params": [param_dict[n] for n in sorted(decay_names)],
            "weight_decay": float(cfg.weight_decay),
        },
        {
            "params": [param_dict[n] for n in sorted(no_decay_names)],
            "weight_decay": 0.0,
        },
    ]
    use_fused = bool(cfg.use_fused_adamw) and str(cfg.device).lower().startswith("cuda")
    if use_fused:
        try:
            optimizer = torch.optim.AdamW(optim_groups, fused=True, **adamw_kwargs)
            log("[optimizer] AdamW fused=True", flush=True)
        except TypeError:
            optimizer = torch.optim.AdamW(optim_groups, **adamw_kwargs)
            log("[optimizer] AdamW fused unsupported; fallback fused=False", flush=True)
    else:
        optimizer = torch.optim.AdamW(optim_groups, **adamw_kwargs)
        log("[optimizer] AdamW fused=False", flush=True)

    def lr_for_step(step: int) -> float:
        base = float(cfg.lr)
        min_lr = float(cfg.lr_min)
        warmup = max(0, int(cfg.lr_warmup_iters))
        total = max(1, int(cfg.max_iters))
        if warmup > 0 and step < warmup:
            return base * (step + 1) / warmup
        if total <= warmup:
            return base
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base - min_lr) * cosine

    writer = None
    if SummaryWriter is not None and rank == 0:
        tb_dir = cfg.tb_logdir if cfg.tb_logdir else str(cfg.out_dir / "runs")
        writer = SummaryWriter(log_dir=tb_dir)

    best_val = math.inf
    start_step = 0
    if cfg.resume is not None:
        resume_path = Path(cfg.resume).expanduser().resolve()
        if not resume_path.exists() or not resume_path.is_file():
            raise SystemExit(f"resume checkpoint not found: {resume_path}")

        ckpt = torch.load(str(resume_path), map_location=cfg.device, weights_only=False)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise SystemExit(f"invalid checkpoint format: {resume_path}")

        ckpt_meta = ckpt.get("meta") if isinstance(ckpt.get("meta"), dict) else {}
        ckpt_vocab = int(ckpt_meta.get("vocab_size", 0) or 0)
        if ckpt_vocab > 0 and ckpt_vocab != int(vocab_size):
            raise SystemExit(
                f"checkpoint vocab_size mismatch: ckpt={ckpt_vocab} current={vocab_size}"
            )

        ckpt_cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
        for key in ("n_layer", "n_head", "n_embd"):
            if key in ckpt_cfg and int(ckpt_cfg.get(key, 0)) != int(getattr(cfg, key)):
                raise SystemExit(
                    f"checkpoint model shape mismatch for {key}: "
                    f"ckpt={int(ckpt_cfg.get(key, 0))} current={int(getattr(cfg, key))}"
                )

        raw_model.load_state_dict(ckpt["model_state"], strict=True)
        opt_state = ckpt.get("optimizer_state")
        if isinstance(opt_state, dict):
            optimizer.load_state_dict(opt_state)
            # Ensure optimizer state tensors are on the correct device —
            # torch may leave them on CPU after load_state_dict in some versions.
            target_dev = torch.device(cfg.device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.device != target_dev:
                        state[k] = v.to(target_dev)
        elif rank == 0:
            log("[resume] optimizer_state missing; continuing with fresh optimizer")

        best_val = float(ckpt.get("best_val_loss", ckpt.get("best_val", math.inf)))
        start_step = max(0, int(ckpt.get("step", -1)) + 1)
        _restore_rng_state(
            ckpt.get("rng_state") if isinstance(ckpt.get("rng_state"), dict) else None,
            device_text=device_text,
            ddp_enabled=ddp_enabled,
            rank=rank,
            log=log,
        )
        log(
            f"[resume] loaded {resume_path} step={start_step} best_val={best_val:.6f}",
            flush=True,
        )

    loop_start = time.time()
    ckpt_interval = int(cfg.checkpoint_interval) or int(cfg.eval_interval)
    if ckpt_interval <= 0:
        ckpt_interval = 1
    if rank == 0 and cfg.show_progress and tqdm is not None and cfg.progress_every <= 0:
        step_iter = tqdm(
            range(start_step, cfg.max_iters + 1),
            total=max(0, cfg.max_iters - start_step + 1),
            desc="Training",
            dynamic_ncols=True,
        )
    else:
        step_iter = range(start_step, cfg.max_iters + 1)

    try:
        for step in step_iter:
            step_lr = lr_for_step(step)
            for pg in optimizer.param_groups:
                pg["lr"] = step_lr

            if step % cfg.eval_interval == 0:
                losses = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    cfg,
                    data_sample_weights,
                    ddp_enabled=ddp_enabled,
                    world_size=world_size,
                    rank=rank,
                )
                log(
                    f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
                )
                if writer is not None:
                    writer.add_scalar("loss/train", losses["train"], step)
                    writer.add_scalar("loss/val", losses["val"], step)
                    writer.add_scalar("lr", step_lr, step)
                if losses["val"] < best_val and rank == 0:
                    best_val = losses["val"]
                    ckpt = _build_checkpoint(
                        raw_model=raw_model,
                        optimizer=optimizer,
                        cfg=cfg,
                        meta=meta,
                        script_flow_version=script_flow_version,
                        tokenizer=tokenizer,
                        best_val=best_val,
                        step=step,
                        ddp_enabled=ddp_enabled,
                        world_size=world_size,
                        device_text=device_text,
                    )
                    torch.save(ckpt, cfg.out_dir / "best.pt")

            optimizer.zero_grad(set_to_none=True)
            accum_steps = max(1, int(cfg.grad_accum_steps))
            for micro_step in range(accum_steps):
                xb, yb = get_batch(
                    train_data,
                    cfg.batch_size,
                    cfg.block_size,
                    torch.device(cfg.device),
                    sample_weights=data_sample_weights,
                )
                amp_ctx = _autocast_ctx(cfg)
                sync_ctx = (
                    model.no_sync()  # type: ignore[attr-defined]
                    if ddp_enabled and micro_step < (accum_steps - 1)
                    else nullcontext()
                )
                with sync_ctx:
                    with amp_ctx:
                        _, loss = model(xb, yb)
                if loss is None:
                    raise RuntimeError("Model returned no loss during training")
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / accum_steps
                loss.backward()
            if float(cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), float(cfg.grad_clip)
                )
            optimizer.step()

            if rank == 0 and step > 0 and step % ckpt_interval == 0:
                ckpt = _build_checkpoint(
                    raw_model=raw_model,
                    optimizer=optimizer,
                    cfg=cfg,
                    meta=meta,
                    script_flow_version=script_flow_version,
                    tokenizer=tokenizer,
                    best_val=best_val,
                    step=step,
                    ddp_enabled=ddp_enabled,
                    world_size=world_size,
                    device_text=device_text,
                )
                torch.save(ckpt, cfg.out_dir / "last_ckpt.pt")

            if cfg.progress_every > 0 and step > 0 and step % cfg.progress_every == 0:
                elapsed_s = max(1e-9, time.time() - loop_start)
                done_steps = max(1, step - start_step + 1)
                rate = done_steps / elapsed_s
                remaining = max(0, cfg.max_iters - step)
                eta_s = remaining / max(rate, 1e-9)
                log(
                    f"[train] step={step}/{cfg.max_iters} | rate={rate:.2f} it/s | elapsed={elapsed_s:.1f}s | eta={eta_s:.1f}s | lr={step_lr:.6g} | grad_accum={cfg.grad_accum_steps}",
                    flush=True,
                )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg and device_text.startswith("cuda"):
            idx = torch.cuda.current_device()
            free_b, total_b = torch.cuda.mem_get_info(idx)
            alloc_b = torch.cuda.memory_allocated(idx)
            reserved_b = torch.cuda.memory_reserved(idx)
            log(
                f"[oom] cuda index={idx} free={free_b / 1024**3:.2f}GiB total={total_b / 1024**3:.2f}GiB allocated={alloc_b / 1024**3:.2f}GiB reserved={reserved_b / 1024**3:.2f}GiB",
                flush=True,
            )
            log(
                "[oom] hint: reduce batch_size/block_size/n_embd/n_layer or use CPU fallback",
                flush=True,
            )
        raise

    if rank == 0:
        final_ckpt = _build_checkpoint(
            raw_model=raw_model,
            optimizer=optimizer,
            cfg=cfg,
            meta=meta,
            script_flow_version=script_flow_version,
            tokenizer=tokenizer,
            best_val=best_val,
            step=int(cfg.max_iters),
            ddp_enabled=ddp_enabled,
            world_size=world_size,
            device_text=device_text,
        )
        torch.save(final_ckpt, cfg.out_dir / "last_ckpt.pt")
        torch.save(raw_model.state_dict(), cfg.out_dir / "last_model_state.pt")

    prompt_text = sanitize_text(cfg.gen_prompt)
    if meta.get("lowercase", False):
        prompt_text = prompt_text.lower()
    prompt_ids = tokenizer.encode(prompt_text).ids
    if not prompt_ids:
        prompt_ids = [meta.get("unk_id", 0)]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)
    if rank == 0:
        out = raw_model.generate(
            context,
            max_new_tokens=cfg.gen_tokens,
            temperature=cfg.gen_temperature,
            top_k=cfg.gen_top_k,
        )[0].tolist()
        sample_text = tokenizer.decode(out, skip_special_tokens=True)
        (cfg.out_dir / "sample.txt").write_text(sample_text, encoding="utf-8")
        log(f"Saved checkpoint: {cfg.out_dir / 'best.pt'}")
        log(f"Saved resume checkpoint: {cfg.out_dir / 'last_ckpt.pt'}")
        log(f"Saved final weights: {cfg.out_dir / 'last_model_state.pt'}")
        log(f"Saved sample generation: {cfg.out_dir / 'sample.txt'}")
    if writer is not None:
        writer.close()
    if ddp_enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _autocast_ctx(cfg: TrainConfig):
    dev = str(cfg.device).lower()
    if dev.startswith("cuda") and str(cfg.precision).lower() == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


if __name__ == "__main__":
    main()
