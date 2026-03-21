from __future__ import annotations

from typing import Any

import torch

from train_model import GPTMini

MODEL_ARCH = "gptmini_v2"


def strip_state_prefixes(state: dict[str, Any]) -> dict[str, Any]:
    def _strip(key: str) -> str:
        out = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if out.startswith(prefix):
                    out = out[len(prefix) :]
                    changed = True
        return out

    return {_strip(str(k)): v for k, v in state.items()}


def detect_model_arch(ckpt: dict[str, Any]) -> str:
    explicit = str(ckpt.get("model_arch", "")).strip()
    if explicit:
        return explicit

    cfg = ckpt.get("config")
    if isinstance(cfg, dict):
        cfg_arch = str(cfg.get("model_arch", "")).strip()
        if cfg_arch:
            return cfg_arch

    return MODEL_ARCH


def build_model_from_ckpt(
    ckpt: dict[str, Any],
    *,
    vocab_size: int,
    device: str,
    gradient_checkpointing: bool = False,
) -> tuple[torch.nn.Module, str]:
    cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    arch = detect_model_arch(ckpt)

    model = GPTMini(
        vocab_size=vocab_size,
        block_size=int(cfg.get("block_size", 256)),
        n_layer=int(cfg.get("n_layer", 8)),
        n_head=int(cfg.get("n_head", 8)),
        n_embd=int(cfg.get("n_embd", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
        gradient_checkpointing=bool(gradient_checkpointing),
    ).to(device)

    raw_state = ckpt.get("model_state")
    if not isinstance(raw_state, dict):
        raise SystemExit("Checkpoint missing valid model_state dict")
    model.load_state_dict(strip_state_prefixes(raw_state), strict=True)
    return model, arch
