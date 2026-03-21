#!/usr/bin/env python3
from __future__ import annotations

import unicodedata


def _strip_control_chars(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch in {"\n", "\t"}:
            out.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        out.append(ch)
    return "".join(out)


def sanitize_text(text: str) -> str:
    """Normalise unicode, strip control chars, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_control_chars(text)
    text = "\n\n".join(part.strip() for part in text.split("\n\n"))
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# Backward-compatible alias
sanitize_english_text = sanitize_text
