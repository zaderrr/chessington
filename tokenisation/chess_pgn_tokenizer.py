#!/usr/bin/env python3
"""
Chess PGN Tokenizer
====================
Tokenizes PGN game files into integer sequences for language model training.

Vocabulary design:
  - Special tokens:  <bos>, <eos>, <unk>, <white_win>, <black_win>, <draw>
  - Move tokens: all unique SAN move strings found in the data
  - Move number tokens: "1.", "2.", ... up to max seen

Usage:
  # Step 1: Build vocabulary from PGN file(s)
  python chess_pgn_tokenizer.py build-vocab --input games.pgn --output vocab.json
  python chess_pgn_tokenizer.py build-vocab --input ./pgn_dir/ --output vocab.json
  python chess_pgn_tokenizer.py build-vocab --input "data/*.pgn" --output vocab.json

  # Step 2: Tokenize PGN files using the vocabulary
  python chess_pgn_tokenizer.py tokenize --input ./pgn_dir/ --vocab vocab.json --output tokens.bin

  # Step 3 (optional): Inspect tokens
  python chess_pgn_tokenizer.py inspect --vocab vocab.json --input tokens.bin --n 5
"""

import argparse
import glob
import json
import re
import sys
import struct
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional


def resolve_pgn_inputs(input_path: str) -> list[str]:
    """
    Resolve --input to a list of .pgn file paths.
    Accepts: a single .pgn file, a directory (scans for *.pgn recursively),
    or a glob pattern.
    """
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        files = sorted(p.rglob("*.pgn"))
        if not files:
            print(f"ERROR: No .pgn files found in {p}")
            sys.exit(1)
        print(f"Found {len(files)} PGN file(s) in {p}")
        return [str(f) for f in files]
    else:
        # Try as glob pattern
        files = sorted(glob.glob(input_path, recursive=True))
        files = [f for f in files if f.endswith(".pgn")]
        if not files:
            print(f"ERROR: No .pgn files matched '{input_path}'")
            sys.exit(1)
        print(f"Matched {len(files)} PGN file(s)")
        return [str(f) for f in files]


# ── Special tokens ──────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    "<bos>",  # 0 - beginning of game
    "<eos>",  # 1 - end of game
    "<unk>",  # 2 - unknown token
    "<white_win>",  # 3 - white wins
    "<black_win>",  # 4 - black wins
    "<draw>",  # 5 - draw
    "*",  # 6 - unknown/ongoing result
]


# ── PGN Parsing ─────────────────────────────────────────────────────────────


def iter_pgn_games(filepath: str):
    """
    Yield raw movetext strings from a PGN file.
    Strips headers, comments, variations, and NAGs.
    Handles large files by streaming line-by-line.
    """
    in_headers = True
    movetext_lines = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                if not in_headers and movetext_lines:
                    # End of a game — yield it
                    raw = " ".join(movetext_lines)
                    cleaned = clean_movetext(raw)
                    if cleaned:
                        yield cleaned
                    movetext_lines = []
                    in_headers = True
                continue

            # Header lines start with [
            if line.startswith("["):
                in_headers = True
                continue

            # Otherwise it's movetext
            in_headers = False
            movetext_lines.append(line)

    # Don't forget the last game if file doesn't end with blank line
    if movetext_lines:
        raw = " ".join(movetext_lines)
        cleaned = clean_movetext(raw)
        if cleaned:
            yield cleaned


def clean_movetext(text: str) -> Optional[str]:
    """Remove comments, variations, NAGs from movetext."""
    # Remove curly-brace comments {like this}
    text = re.sub(r"\{[^}]*\}", "", text)
    # Remove parenthesized variations (non-recursive, handles one level)
    text = re.sub(r"\([^)]*\)", "", text)
    # Remove NAGs like $1, $2, etc.
    text = re.sub(r"\$\d+", "", text)
    # Remove semicolon comments (rest of line, but we've joined lines)
    text = re.sub(r";[^\n]*", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def tokenize_movetext(text: str) -> list[str]:
    """
    Split cleaned movetext into tokens.
    Produces: ["1.", "e4", "e5", "2.", "Nf3", "Nc6", ..., "1-0"]

    Move numbers become their own token ("1.", "2.", etc.)
    Each move (SAN notation) becomes its own token.
    Result at the end becomes a token.
    """
    tokens = []
    # Pattern matches: move numbers (1. or 1...), SAN moves, results
    # SAN moves: optional piece letter, optional file/rank disambiguation,
    #   optional capture 'x', destination square, optional promotion, optional +/#
    # Also handles O-O, O-O-O castling
    pattern = re.compile(
        r"(\d+\.{1,3})"  # move number: "1." or "1..."
        r"|([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)"  # standard SAN
        r"|(O-O-O[+#]?|O-O[+#]?)"  # castling
        r"|(1-0|0-1|1/2-1/2|\*)"  # result
    )

    for m in pattern.finditer(text):
        token = m.group(0)
        # Normalize move numbers: "1..." -> "1."  (black's move after ellipsis)
        if re.match(r"\d+\.{2,}", token):
            token = token.rstrip(".") + "."
        tokens.append(token)

    return tokens


# ── Vocabulary ──────────────────────────────────────────────────────────────


class ChessVocab:
    """Token <-> ID mapping for chess PGN tokens."""

    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    @classmethod
    def build(
        cls, pgn_paths: list[str], max_games: int = 0, min_freq: int = 1
    ) -> "ChessVocab":
        """Build vocabulary by scanning PGN files."""
        vocab = cls()
        counts: Counter = Counter()
        n_games = 0

        for pgn_path in pgn_paths:
            print(f"Scanning {pgn_path} ...")
            for movetext in iter_pgn_games(pgn_path):
                tokens = tokenize_movetext(movetext)
                counts.update(tokens)
                n_games += 1
                if n_games % 100_000 == 0:
                    print(
                        f"  ...scanned {n_games:,} games, {len(counts):,} unique tokens"
                    )
                if max_games and n_games >= max_games:
                    break
            if max_games and n_games >= max_games:
                break

        print(f"Scanned {n_games:,} games total, {len(counts):,} unique tokens found")

        # Add special tokens first
        for st in SPECIAL_TOKENS:
            idx = len(vocab.token_to_id)
            vocab.token_to_id[st] = idx
            vocab.id_to_token[idx] = st

        # Add move number tokens in order (1. through max seen)
        move_nums = sorted(
            [t for t in counts if re.match(r"^\d+\.$", t)], key=lambda x: int(x[:-1])
        )
        for mn in move_nums:
            if mn not in vocab.token_to_id:
                idx = len(vocab.token_to_id)
                vocab.token_to_id[mn] = idx
                vocab.id_to_token[idx] = mn

        # Add all other tokens sorted by frequency (descending)
        for token, freq in counts.most_common():
            if freq < min_freq:
                continue
            if token not in vocab.token_to_id:
                idx = len(vocab.token_to_id)
                vocab.token_to_id[token] = idx
                vocab.id_to_token[idx] = token

        print(f"Vocabulary size: {len(vocab.token_to_id):,} tokens")
        return vocab

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert token strings to IDs."""
        unk_id = self.token_to_id["<unk>"]
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Convert IDs back to token strings."""
        return [self.id_to_token.get(i, "<unk>") for i in ids]

    def save(self, path: str):
        """Save vocabulary to JSON."""
        data = {
            "version": 1,
            "size": len(self.token_to_id),
            "token_to_id": self.token_to_id,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved vocabulary ({len(self.token_to_id):,} tokens) to {path}")

    @classmethod
    def load(cls, path: str) -> "ChessVocab":
        """Load vocabulary from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        vocab = cls()
        vocab.token_to_id = data["token_to_id"]
        vocab.id_to_token = {int(v): k for k, v in vocab.token_to_id.items()}
        print(f"Loaded vocabulary ({len(vocab.token_to_id):,} tokens) from {path}")
        return vocab

    @property
    def size(self) -> int:
        return len(self.token_to_id)


# ── Tokenization to binary ─────────────────────────────────────────────────


def tokenize_to_bin(
    pgn_paths: list[str],
    vocab: ChessVocab,
    output_path: str,
    max_games: int = 0,
    dtype: str = "uint16",
):
    """
    Tokenize PGN file(s) into a flat numpy memmap .bin file.

    Each game becomes: <bos> [move tokens...] [result] <eos>
    Games are concatenated into a single flat array (GPT-style).

    Uses uint16 by default — supports vocab up to 65,535 tokens.
    """
    np_dtype = np.uint16 if dtype == "uint16" else np.uint32

    if vocab.size > 65535 and dtype == "uint16":
        print("WARNING: Vocab size exceeds uint16 range, switching to uint32")
        np_dtype = np.uint32

    bos_id = vocab.token_to_id["<bos>"]
    eos_id = vocab.token_to_id["<eos>"]

    # First pass: count total tokens to allocate memmap
    print("Pass 1/2: counting tokens...")
    total_tokens = 0
    n_games = 0
    n_unk = 0
    unk_id = vocab.token_to_id["<unk>"]

    for pgn_path in pgn_paths:
        for movetext in iter_pgn_games(pgn_path):
            tokens = tokenize_movetext(movetext)
            if not tokens:
                continue
            total_tokens += len(tokens) + 2  # +2 for <bos> and <eos>
            n_games += 1
            if max_games and n_games >= max_games:
                break
        if max_games and n_games >= max_games:
            break

    print(f"  {n_games:,} games, {total_tokens:,} total tokens")

    # Second pass: write tokens
    print(f"Pass 2/2: writing {output_path}...")
    arr = np.memmap(output_path, dtype=np_dtype, mode="w+", shape=(total_tokens,))

    offset = 0
    n_games = 0
    for pgn_path in pgn_paths:
        print(f"  Processing {pgn_path} ...")
        for movetext in iter_pgn_games(pgn_path):
            tokens = tokenize_movetext(movetext)
            if not tokens:
                continue

            ids = vocab.encode(tokens)
            n_unk += ids.count(unk_id)

            game_ids = [bos_id] + ids + [eos_id]
            arr[offset : offset + len(game_ids)] = game_ids
            offset += len(game_ids)

            n_games += 1
            if n_games % 100_000 == 0:
                print(f"  ...wrote {n_games:,} games ({offset:,} tokens)")
            if max_games and n_games >= max_games:
                break
        if max_games and n_games >= max_games:
            break

    arr.flush()
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Done! {n_games:,} games, {total_tokens:,} tokens, {file_size_mb:.1f} MB")
    if n_unk > 0:
        print(
            f"  WARNING: {n_unk:,} unknown tokens encountered ({n_unk / total_tokens * 100:.2f}%)"
        )


# ── Inspect ─────────────────────────────────────────────────────────────────


def inspect_bin(vocab: ChessVocab, bin_path: str, n_games: int = 5):
    """Print the first N games from a tokenized .bin file."""
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    print(f"Total tokens in file: {len(arr):,}\n")

    bos_id = vocab.token_to_id["<bos>"]
    eos_id = vocab.token_to_id["<eos>"]

    games_shown = 0
    game_start = None

    for i, tok_id in enumerate(arr):
        if tok_id == bos_id:
            game_start = i
        elif tok_id == eos_id and game_start is not None:
            game_ids = arr[game_start : i + 1].tolist()
            game_tokens = vocab.decode(game_ids)
            print(f"Game {games_shown + 1} ({len(game_ids)} tokens):")
            # Pretty print: skip <bos>, join moves with spaces
            moves = game_tokens[1:-1]  # strip <bos> and <eos>
            print(f"  {' '.join(moves)}")
            print()
            games_shown += 1
            game_start = None
            if games_shown >= n_games:
                break

    # Vocab stats
    print(f"--- Vocab stats ---")
    print(f"  Total vocab size: {vocab.size}")
    n_special = len(SPECIAL_TOKENS)
    n_movenums = sum(1 for t in vocab.token_to_id if re.match(r"^\d+\.$", t))
    n_moves = vocab.size - n_special - n_movenums
    print(f"  Special tokens: {n_special}")
    print(f"  Move number tokens: {n_movenums}")
    print(f"  Unique move tokens: {n_moves}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Chess PGN Tokenizer for LM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-vocab
    p_vocab = sub.add_parser("build-vocab", help="Build vocabulary from PGN file(s)")
    p_vocab.add_argument(
        "--input", "-i", required=True, help="PGN file, directory, or glob pattern"
    )
    p_vocab.add_argument("--output", "-o", required=True, help="Output vocab JSON file")
    p_vocab.add_argument(
        "--max-games", type=int, default=0, help="Max games to scan (0=all)"
    )
    p_vocab.add_argument(
        "--min-freq", type=int, default=1, help="Min token frequency to include"
    )

    # tokenize
    p_tok = sub.add_parser("tokenize", help="Tokenize PGN file(s) to .bin")
    p_tok.add_argument(
        "--input", "-i", required=True, help="PGN file, directory, or glob pattern"
    )
    p_tok.add_argument("--vocab", "-v", required=True, help="Vocabulary JSON file")
    p_tok.add_argument("--output", "-o", required=True, help="Output .bin file")
    p_tok.add_argument(
        "--max-games", type=int, default=0, help="Max games to tokenize (0=all)"
    )
    p_tok.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")

    # inspect
    p_insp = sub.add_parser("inspect", help="Inspect tokenized .bin file")
    p_insp.add_argument("--input", "-i", required=True, help="Input .bin file")
    p_insp.add_argument("--vocab", "-v", required=True, help="Vocabulary JSON file")
    p_insp.add_argument("--n", type=int, default=5, help="Number of games to show")

    args = parser.parse_args()

    if args.command == "build-vocab":
        pgn_files = resolve_pgn_inputs(args.input)
        vocab = ChessVocab.build(
            pgn_files, max_games=args.max_games, min_freq=args.min_freq
        )
        vocab.save(args.output)

    elif args.command == "tokenize":
        pgn_files = resolve_pgn_inputs(args.input)
        vocab = ChessVocab.load(args.vocab)
        tokenize_to_bin(
            pgn_files, vocab, args.output, max_games=args.max_games, dtype=args.dtype
        )

    elif args.command == "inspect":
        vocab = ChessVocab.load(args.vocab)
        inspect_bin(vocab, args.input, n_games=args.n)


if __name__ == "__main__":
    main()
