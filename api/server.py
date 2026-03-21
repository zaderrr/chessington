import asyncio
import atexit
from datetime import datetime
import importlib
import json
import os
import re
import subprocess
import sys
import threading
from io import StringIO
from pathlib import Path
from typing import Optional

serve = importlib.import_module("websockets.asyncio.server").serve


def load_local_config() -> dict:
    config_path = Path(
        os.getenv("CHESSINGTON_CONFIG", Path(__file__).with_name("server.config.json"))
    ).expanduser()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return data


def default_device() -> str:
    try:
        torch_mod = importlib.import_module("torch")
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


CONFIG = load_local_config()

chess = importlib.import_module("chess")
chess_pgn = importlib.import_module("chess.pgn")


def first_move_token(raw: str) -> Optional[str]:
    without_headers = re.sub(r"^\s*(?:\[[^\]]*\]\s*)*", "", raw).strip()
    without_leading_move_number = re.sub(r"^\d+\.(?:\.\.)?\s*", "", without_headers)
    match = re.match(r"^([^\s{}()]+)", without_leading_move_number)
    if not match:
        return None
    token = match.group(1)
    if token in {"*", "1-0", "0-1", "1/2-1/2"}:
        return None
    return token


def board_from_movetext(movetext: str):
    pgn_source = StringIO(f'[Event "Live"]\n\n{movetext.strip() if movetext else ""}')
    game = chess_pgn.read_game(pgn_source)
    if game is None:
        return chess.Board()

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def legalize_move(candidate: str, board) -> Optional[str]:
    token = first_move_token(candidate)
    if not token:
        return None

    try:
        parsed_san = board.parse_san(token)
        if parsed_san in board.legal_moves:
            return token
    except Exception:
        pass

    if re.match(r"^[a-h][1-8][a-h][1-8][qrbn]?$", token.lower()):
        try:
            parsed_uci = chess.Move.from_uci(token.lower())
            if parsed_uci in board.legal_moves:
                return token.lower()
        except Exception:
            pass

    return None


def pgn_archive_dir() -> Path:
    configured = os.getenv("CHESSINGTON_PGN_DIR") or CONFIG.get(
        "pgn_archive_dir", "data/games"
    )
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    configured = os.getenv("CHESSINGTON_MODELS_DIR") or CONFIG.get(
        "models_dir", "models"
    )
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pgn_file(pgn: str, white: str, black: str) -> Path:
    safe_white = re.sub(r"[^a-zA-Z0-9_-]+", "-", white.strip() or "white").strip("-")
    safe_black = re.sub(r"[^a-zA-Z0-9_-]+", "-", black.strip() or "black").strip("-")
    if not safe_white:
        safe_white = "white"
    if not safe_black:
        safe_black = "black"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{safe_white}_vs_{safe_black}.pgn"
    output_path = pgn_archive_dir() / filename
    output_path.write_text(pgn.strip() + "\n", encoding="utf-8")
    return output_path


def invalid_move_log_path() -> Path:
    configured = os.getenv("CHESSINGTON_INVALID_MOVE_LOG") or CONFIG.get(
        "invalid_move_log", "data/chessington_invalid_moves.jsonl"
    )
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def log_invalid_move_attempt(
    movetext: str,
    candidate: str,
    attempt: int,
    max_attempts: int,
    reason: str,
    temperature: float,
) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "attempt": attempt,
        "max_attempts": max_attempts,
        "candidate": candidate,
        "temperature": temperature,
        "reason": reason,
        "movetext": movetext,
    }
    with invalid_move_log_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


class InferenceEngine:
    def __init__(self) -> None:
        training_dir = os.getenv("CHESSINGTON_TRAINING_DIR") or CONFIG.get(
            "training_dir", ""
        )
        if training_dir:
            td = Path(str(training_dir)).expanduser()
            if not td.is_absolute():
                td = (Path(__file__).resolve().parent.parent / td).resolve()
            self.training_dir = td
        else:
            self.training_dir = None

        ckpt_value = os.getenv("CHESSINGTON_CKPT") or CONFIG.get("checkpoint_path", "")
        self.ckpt_path = Path(str(ckpt_value)).expanduser() if ckpt_value else None
        self.models_root = models_dir()
        self.model_name = ""

        self.device = os.getenv("CHESSINGTON_DEVICE") or str(
            CONFIG.get("device", default_device())
        )
        self.max_new_tokens = int(
            os.getenv("CHESSINGTON_MAX_NEW_TOKENS") or CONFIG.get("max_new_tokens", 128)
        )
        self.temperature = float(
            os.getenv("CHESSINGTON_TEMPERATURE") or CONFIG.get("temperature", 0.8)
        )
        self.top_k = int(os.getenv("CHESSINGTON_TOP_K") or CONFIG.get("top_k", 50))
        self.model = None
        self.tokenizer = None
        self.meta = None
        self.loaded_ckpt_meta = {}
        self.is_sft_model = False
        self.sft_white_win_token = str(
            os.getenv("CHESSINGTON_SFT_WHITE_WIN_TOKEN")
            or CONFIG.get("sft_white_win_token", "<white_win>")
        )
        self.sft_black_win_token = str(
            os.getenv("CHESSINGTON_SFT_BLACK_WIN_TOKEN")
            or CONFIG.get("sft_black_win_token", "<black_win>")
        )
        self._run_once = None
        self._lock = threading.Lock()
        self.retry_attempts = int(
            os.getenv("CHESSINGTON_MOVE_RETRIES") or CONFIG.get("move_retries", 5)
        )
        self.retry_temperature_step = float(
            os.getenv("CHESSINGTON_RETRY_TEMPERATURE_STEP")
            or CONFIG.get("retry_temperature_step", 0.2)
        )
        self.max_retry_temperature = float(
            os.getenv("CHESSINGTON_MAX_RETRY_TEMPERATURE")
            or CONFIG.get("max_retry_temperature", 1.6)
        )

    def _discover_model_options(self):
        options: dict[str, dict] = {}

        configured_models = CONFIG.get("models", [])
        if isinstance(configured_models, list):
            for item in configured_models:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                path_raw = str(item.get("checkpoint_path", "")).strip()
                if not name or not path_raw:
                    continue
                path = Path(path_raw).expanduser()
                if not path.is_absolute():
                    path = Path(__file__).resolve().parents[1] / path
                if path.exists():
                    is_sft = bool(item.get("is_sft", False)) or ("sft" in name.lower())
                    options[name] = {"path": path, "is_sft": is_sft}

        for ckpt in sorted(self.models_root.glob("*/best.pt")):
            name = ckpt.parent.name
            options.setdefault(name, {"path": ckpt, "is_sft": "sft" in name.lower()})

        if self.ckpt_path is not None and self.ckpt_path.exists():
            fallback_name = self.ckpt_path.parent.name or "configured"
            options.setdefault(
                fallback_name,
                {"path": self.ckpt_path, "is_sft": "sft" in fallback_name.lower()},
            )

        return [
            {
                "name": name,
                "checkpoint_path": str(entry["path"]),
                "is_sft": bool(entry.get("is_sft", False)),
            }
            for name, entry in sorted(options.items(), key=lambda item: item[0].lower())
        ]

    def available_models(self) -> dict:
        models = self._discover_model_options()
        current_model = self.model_name
        if not current_model and self.ckpt_path is not None:
            current_model = self.ckpt_path.parent.name
        return {
            "models": models,
            "current_model": current_model,
            "current_checkpoint": str(self.ckpt_path) if self.ckpt_path else "",
            "current_is_sft": bool(self.is_sft_model),
        }

    def set_model(self, model_name: str) -> dict:
        name = str(model_name or "").strip()
        if not name:
            raise ValueError("Model name is required")

        options = self._discover_model_options()
        selected = next((x for x in options if x["name"] == name), None)
        if selected is None:
            available = ", ".join(x["name"] for x in options) or "none"
            raise ValueError(f"Unknown model '{name}'. Available: {available}")

        with self._lock:
            self.ckpt_path = Path(selected["checkpoint_path"])
            self.model_name = selected["name"]
            self.model = None
            self.tokenizer = None
            self.meta = None
            self.loaded_ckpt_meta = {}
            self.is_sft_model = False

        return {
            "current_model": self.model_name,
            "current_checkpoint": str(self.ckpt_path),
            "current_is_sft": bool(self.is_sft_model),
        }

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            if self.training_dir is None:
                raise ValueError(
                    "Training directory is not configured. Add training_dir to api/server.config.json "
                    "or set CHESSINGTON_TRAINING_DIR."
                )
            if str(self.training_dir) not in sys.path:
                sys.path.insert(0, str(self.training_dir))

            try:
                infer_mod = importlib.import_module("infer")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to import inference modules from {self.training_dir}: {exc}"
                ) from exc

            load_checkpoint = infer_mod.load_checkpoint
            build_model_from_checkpoint = infer_mod.build_model_from_checkpoint
            self._run_once = infer_mod.run_once

            if self.ckpt_path is None:
                options = self._discover_model_options()
                if not options:
                    raise FileNotFoundError(
                        "No models found. Place checkpoints under models/<name>/best.pt "
                        "or configure checkpoint_path in api/server.config.json"
                    )
                self.model_name = options[0]["name"]
                self.ckpt_path = Path(options[0]["checkpoint_path"])

            if not self.ckpt_path.exists():
                raise FileNotFoundError(
                    "Checkpoint not found. Set checkpoint_path in api/server.config.json, "
                    "set CHESSINGTON_CKPT, or place model at "
                    f"{self.ckpt_path}"
                )

            if not self.model_name:
                self.model_name = self.ckpt_path.parent.name

            ckpt = load_checkpoint(self.ckpt_path, self.device)
            self.loaded_ckpt_meta = ckpt
            self.is_sft_model = bool(ckpt.get("sft")) or (
                "sft" in self.model_name.lower()
            )
            self.model, self.tokenizer, self.meta, _ = build_model_from_checkpoint(
                ckpt, self.device
            )

    def infer(self, prompt: str, temperature: Optional[float] = None) -> str:
        self._ensure_loaded()
        if self._run_once is None:
            raise RuntimeError("Inference function not initialized")
        selected_temperature = (
            temperature if temperature is not None else self.temperature
        )
        output = self._run_once(
            self.model,
            self.tokenizer,
            self.meta,
            prompt,
            self.max_new_tokens,
            selected_temperature,
            self.top_k,
            self.device,
        )
        print(output)
        return output

    def infer_legal_move(self, movetext: str) -> str:
        board = board_from_movetext(movetext)
        prompt = movetext
        if self.is_sft_model:
            side_token = (
                self.sft_white_win_token
                if board.turn == chess.WHITE
                else self.sft_black_win_token
            )
            prompt = f"{side_token} {movetext}".strip()

        attempts = max(1, self.retry_attempts)
        for attempt in range(1, attempts + 1):
            retry_temperature = min(
                self.temperature + ((attempt - 1) * self.retry_temperature_step),
                self.max_retry_temperature,
            )
            candidate = self.infer(prompt, retry_temperature)
            legal_move = legalize_move(candidate, board)
            if legal_move:
                return legal_move
            log_invalid_move_attempt(
                prompt,
                candidate,
                attempt,
                attempts,
                "illegal_or_unparseable_move",
                retry_temperature,
            )

        raise RuntimeError(
            f"Chessington did not return a legal move after {attempts} attempts"
        )


class StockfishEngine:
    def __init__(self) -> None:
        stockfish_value = os.getenv("CHESSINGTON_STOCKFISH_BIN") or CONFIG.get(
            "stockfish_path", "../stockfish/stockfish"
        )
        if stockfish_value:
            self.binary_path = Path(str(stockfish_value)).expanduser()
        else:
            self.binary_path = (
                Path(__file__).resolve().parents[1] / "stockfish" / "stockfish"
            )
        self.movetime_ms = int(
            os.getenv("CHESSINGTON_STOCKFISH_MOVETIME")
            or CONFIG.get("stockfish_movetime_ms", 350)
        )
        self.default_nodes = int(
            os.getenv("CHESSINGTON_STOCKFISH_NODES") or CONFIG.get("stockfish_nodes", 0)
        )
        self.default_depth = int(
            os.getenv("CHESSINGTON_STOCKFISH_DEPTH") or CONFIG.get("stockfish_depth", 0)
        )
        self.process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Stockfish process is not running")
        self.process.stdin.write(f"{command}\n")
        self.process.stdin.flush()

    def _read_until(self, target_prefix: str) -> str:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Stockfish process is not running")

        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish stopped unexpectedly")
            stripped = line.strip()
            if stripped.startswith(target_prefix):
                return stripped

    def _ensure_started(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        if not self.binary_path.exists():
            raise FileNotFoundError(
                "Stockfish binary not found. Set stockfish_path in api/server.config.json, "
                "set CHESSINGTON_STOCKFISH_BIN, or place binary at "
                f"{self.binary_path}"
            )

        self.process = subprocess.Popen(
            [str(self.binary_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("isready")
        self._read_until("readyok")

    def bestmove(
        self,
        fen: str,
        movetime_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        depth: Optional[int] = None,
    ) -> str:
        with self._lock:
            self._ensure_started()
            self._send(f"position fen {fen}")

            go_parts = ["go"]
            selected_nodes = nodes if nodes is not None else self.default_nodes
            selected_depth = depth if depth is not None else self.default_depth
            selected_movetime = (
                movetime_ms if movetime_ms is not None else self.movetime_ms
            )

            if selected_nodes > 0:
                go_parts.extend(["nodes", str(selected_nodes)])
            if selected_depth > 0:
                go_parts.extend(["depth", str(selected_depth)])
            if selected_movetime > 0:
                go_parts.extend(["movetime", str(selected_movetime)])
            if len(go_parts) == 1:
                go_parts.extend(["movetime", str(self.movetime_ms)])

            self._send(" ".join(go_parts))
            line = self._read_until("bestmove")

        parts = line.split()
        if len(parts) < 2 or parts[1] == "(none)":
            raise RuntimeError("Stockfish did not return a legal move")
        return parts[1]

    def close(self) -> None:
        with self._lock:
            if self.process is None:
                return
            try:
                self._send("quit")
            except Exception:
                pass
            self.process.terminate()
            self.process = None


engine = InferenceEngine()
stockfish_engine = StockfishEngine()
atexit.register(stockfish_engine.close)
HOST = str(os.getenv("CHESSINGTON_HOST") or CONFIG.get("host", "localhost"))
PORT = int(os.getenv("CHESSINGTON_PORT") or CONFIG.get("port", 8765))


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmarks_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def puzzles_path() -> Path:
    return Path(__file__).resolve().parents[1] / "benchmarks" / "puzzles" / "puzzles.json"


def load_puzzles() -> list:
    path = puzzles_path()
    if not path.exists():
        raise FileNotFoundError(f"Puzzles file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("puzzles.json must be a JSON array")
    return data


def _extract_movetext_to_fen(pgn_text: str, target_fen: str) -> Optional[str]:
    """Parse a full PGN and return movetext up to (not including) the target FEN."""
    game = chess_pgn.read_game(StringIO(pgn_text))
    if game is None:
        return None

    target_parts = target_fen.split()[:4]
    board = game.board()

    if board.fen().split()[:4] == target_parts:
        return ""

    collected_moves: list[tuple] = []
    node = game
    while node.variations:
        next_node = node.variations[0]
        collected_moves.append((board.copy(), next_node.move))
        board.push(next_node.move)
        if board.fen().split()[:4] == target_parts:
            break
        node = next_node
    else:
        return None

    parts: list[str] = []
    for b, m in collected_moves:
        san = b.san(m)
        if b.turn == chess.WHITE:
            parts.append(f"{b.fullmove_number}. {san}")
        else:
            if not parts:
                parts.append(f"{b.fullmove_number}... {san}")
            else:
                parts.append(san)
    return " ".join(parts)


def _append_move_san(movetext: str, board, move) -> str:
    """Append a move to PGN movetext with proper numbering. Board is BEFORE the move."""
    san = board.san(move)
    if board.turn == chess.WHITE:
        prefix = f"{board.fullmove_number}."
        if movetext:
            return f"{movetext} {prefix} {san}"
        return f"{prefix} {san}"
    else:
        if not movetext:
            return f"{board.fullmove_number}... {san}"
        return f"{movetext} {san}"


def _parse_predicted_move(predicted: str, board) -> Optional[object]:
    """Try to parse a predicted move string as a chess.Move on the given board."""
    try:
        return board.parse_san(predicted)
    except Exception:
        pass
    if re.match(r"^[a-h][1-8][a-h][1-8][qrbn]?$", predicted.lower()):
        try:
            move = chess.Move.from_uci(predicted.lower())
            if move in board.legal_moves:
                return move
        except Exception:
            pass
    return None


def _moves_match(predicted: Optional[str], expected_uci: str, board) -> bool:
    """Check if a predicted move matches the expected move, or is an
    equally valid checkmate when multiple mating moves exist."""
    if not predicted:
        return False
    try:
        expected_move = chess.Move.from_uci(expected_uci)
    except Exception:
        return False

    predicted_move = _parse_predicted_move(predicted, board)
    if predicted_move is None:
        return False
    if predicted_move == expected_move:
        return True

    # Accept alternative checkmates: if the expected move is mate and
    # the predicted move is also mate, count it as correct.
    board.push(expected_move)
    expected_is_mate = board.is_checkmate()
    board.pop()
    if expected_is_mate:
        board.push(predicted_move)
        predicted_is_mate = board.is_checkmate()
        board.pop()
        if predicted_is_mate:
            return True

    return False


def process_puzzle(puzzle: dict, inf_engine: InferenceEngine) -> dict:
    """Run a single puzzle through the engine, returning per-move results."""
    fen = puzzle["Starting position"]
    uci_moves = puzzle["Moves"].split()
    pgn_text = puzzle.get("PGN", "")

    context = _extract_movetext_to_fen(pgn_text, fen)
    if context is None:
        context = ""

    board = chess.Board(fen)
    movetext = context

    setup_uci = uci_moves[0]
    setup_move = chess.Move.from_uci(setup_uci)
    setup_san = board.san(setup_move)
    setup_side = "w" if board.turn == chess.WHITE else "b"
    movetext = _append_move_san(movetext, board, setup_move)
    board.push(setup_move)

    move_results: list[dict] = []
    full_sequence: list[dict] = [
        {
            "san": setup_san,
            "uci": setup_uci,
            "side": setup_side,
            "role": "setup",
            "fen_after": board.fen(),
        }
    ]

    for i in range(1, len(uci_moves)):
        expected_uci = uci_moves[i]
        expected_move = chess.Move.from_uci(expected_uci)
        expected_san = board.san(expected_move)
        move_side = "w" if board.turn == chess.WHITE else "b"

        if i % 2 == 1:
            predicted = None
            correct = False
            try:
                prompt = movetext
                if inf_engine.is_sft_model:
                    side_token = (
                        inf_engine.sft_white_win_token
                        if board.turn == chess.WHITE
                        else inf_engine.sft_black_win_token
                    )
                    prompt = f"{side_token} {movetext}".strip()
                raw = inf_engine.infer(prompt)
                predicted = first_move_token(raw)
                correct = _moves_match(predicted, expected_uci, board)
            except Exception as exc:
                predicted = f"[error: {exc}]"
                correct = False

            move_results.append(
                {
                    "expected": expected_san,
                    "expected_uci": expected_uci,
                    "predicted": predicted,
                    "correct": correct,
                }
            )

        movetext = _append_move_san(movetext, board, expected_move)
        board.push(expected_move)

        full_sequence.append(
            {
                "san": expected_san,
                "uci": expected_uci,
                "side": move_side,
                "role": "prediction" if i % 2 == 1 else "response",
                "fen_after": board.fen(),
                "predicted": move_results[-1]["predicted"] if i % 2 == 1 else None,
                "correct": move_results[-1]["correct"] if i % 2 == 1 else None,
            }
        )

    total = len(move_results)
    correct_count = sum(1 for r in move_results if r["correct"])
    return {
        "id": puzzle.get("ID", ""),
        "rating": int(puzzle.get("Rating", 0)),
        "themes": puzzle.get("Themes", []),
        "fen": fen,
        "total_predictions": total,
        "correct_predictions": correct_count,
        "depth": f"{correct_count}/{total}",
        "moves": move_results,
        "sequence": full_sequence,
    }


def _benchmark_summary(results: list[dict]) -> dict:
    total_puzzles = len(results)
    if total_puzzles == 0:
        return {
            "total_puzzles": 0,
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0,
            "fully_solved": 0,
            "fully_solved_pct": 0,
            "first_move_correct": 0,
            "first_move_pct": 0,
            "by_rating": {},
            "by_theme": {},
        }

    total_predictions = sum(r["total_predictions"] for r in results)
    correct_predictions = sum(r["correct_predictions"] for r in results)
    fully_solved = sum(
        1 for r in results if r["correct_predictions"] == r["total_predictions"]
    )
    first_move_correct = sum(
        1
        for r in results
        if r["moves"] and r["moves"][0]["correct"]
    )

    by_rating: dict[int, dict] = {}
    for r in results:
        bracket = (r["rating"] // 200) * 200
        bucket = by_rating.setdefault(
            bracket,
            {"count": 0, "correct": 0, "total_predictions": 0, "correct_predictions": 0},
        )
        bucket["count"] += 1
        bucket["correct"] += 1 if r["correct_predictions"] == r["total_predictions"] else 0
        bucket["total_predictions"] += r["total_predictions"]
        bucket["correct_predictions"] += r["correct_predictions"]

    by_rating_labeled = {
        f"{k}-{k + 199}": v for k, v in sorted(by_rating.items())
    }

    by_theme: dict[str, dict] = {}
    for r in results:
        for theme in r.get("themes", []):
            if not theme:
                continue
            bucket = by_theme.setdefault(
                theme,
                {"count": 0, "correct": 0, "total_predictions": 0, "correct_predictions": 0},
            )
            bucket["count"] += 1
            bucket["correct"] += 1 if r["correct_predictions"] == r["total_predictions"] else 0
            bucket["total_predictions"] += r["total_predictions"]
            bucket["correct_predictions"] += r["correct_predictions"]

    return {
        "total_puzzles": total_puzzles,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": round(correct_predictions / total_predictions * 100, 1) if total_predictions else 0,
        "fully_solved": fully_solved,
        "fully_solved_pct": round(fully_solved / total_puzzles * 100, 1),
        "first_move_correct": first_move_correct,
        "first_move_pct": round(first_move_correct / total_puzzles * 100, 1),
        "by_rating": by_rating_labeled,
        "by_theme": dict(sorted(by_theme.items())),
    }


_benchmark_cancel = threading.Event()


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def echo(websocket):
    global _benchmark_cancel
    benchmark_task: Optional[asyncio.Task] = None

    async def run_benchmark(puzzle_count: Optional[int]):
        _benchmark_cancel.clear()
        puzzles = load_puzzles()
        if puzzle_count and puzzle_count > 0:
            puzzles = puzzles[:puzzle_count]

        model_name = engine.model_name or (
            engine.ckpt_path.parent.name if engine.ckpt_path else "unknown"
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_file = benchmarks_dir() / f"{model_name}_{timestamp}.json"

        progress = {
            "model": model_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "puzzle_count": len(puzzles),
            "completed": 0,
            "results": [],
        }

        for i, puzzle in enumerate(puzzles):
            if _benchmark_cancel.is_set():
                break

            result = await asyncio.to_thread(process_puzzle, puzzle, engine)
            progress["results"].append(result)
            progress["completed"] = i + 1

            if (i + 1) % 5 == 0 or (i + 1) == len(puzzles):
                progress["finished_at"] = datetime.now().isoformat(timespec="seconds")
                results_file.write_text(
                    json.dumps(progress, indent=2), encoding="utf-8"
                )

            await websocket.send(
                json.dumps(
                    {
                        "type": "benchmark_progress",
                        "completed": i + 1,
                        "total": len(puzzles),
                        "latest_result": result,
                        "summary": _benchmark_summary(progress["results"]),
                        "current_fen": puzzle.get("Starting position", ""),
                    }
                )
            )

        progress["finished_at"] = datetime.now().isoformat(timespec="seconds")
        results_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")

        await websocket.send(
            json.dumps(
                {
                    "type": "benchmark_complete",
                    "completed": progress["completed"],
                    "total": len(puzzles),
                    "summary": _benchmark_summary(progress["results"]),
                    "results_file": str(results_file),
                    "cancelled": _benchmark_cancel.is_set(),
                }
            )
        )

    async for message in websocket:
        try:
            payload = json.loads(message)
            if isinstance(payload, dict):
                action = payload.get("action")
                if action == "chessington_move":
                    pgn = str(payload.get("pgn", "")).strip()
                    if not pgn:
                        raise ValueError("Missing pgn")
                    move_text = await asyncio.to_thread(engine.infer_legal_move, pgn)
                    san_move = move_text.strip() if move_text else ""
                    response = json.dumps(
                        {"type": "chessington_move", "move": san_move}
                    )
                elif action == "stockfish_bestmove":
                    fen = str(payload.get("fen", "")).strip()
                    if not fen:
                        raise ValueError("Missing fen")
                    movetime = payload.get("movetime")
                    movetime_int = int(movetime) if movetime is not None else None
                    nodes = payload.get("nodes")
                    nodes_int = int(nodes) if nodes is not None else None
                    depth = payload.get("depth")
                    depth_int = int(depth) if depth is not None else None
                    best_move = await asyncio.to_thread(
                        stockfish_engine.bestmove,
                        fen,
                        movetime_int,
                        nodes_int,
                        depth_int,
                    )
                    response = json.dumps(
                        {"type": "stockfish_bestmove", "move": best_move}
                    )
                elif action == "save_game_pgn":
                    pgn = str(payload.get("pgn", "")).strip()
                    if not pgn:
                        raise ValueError("Missing pgn")
                    white = str(payload.get("white", "White"))
                    black = str(payload.get("black", "Black"))
                    saved_path = await asyncio.to_thread(
                        save_pgn_file, pgn, white, black
                    )
                    response = json.dumps(
                        {
                            "type": "game_saved",
                            "path": str(saved_path),
                        }
                    )
                elif action == "list_models":
                    data = await asyncio.to_thread(engine.available_models)
                    response = json.dumps(
                        {
                            "type": "model_list",
                            "models": data.get("models", []),
                            "current_model": data.get("current_model", ""),
                            "current_checkpoint": data.get("current_checkpoint", ""),
                            "current_is_sft": data.get("current_is_sft", False),
                        }
                    )
                elif action == "set_model":
                    name = str(payload.get("model_name", "")).strip()
                    selected = await asyncio.to_thread(engine.set_model, name)
                    response = json.dumps(
                        {
                            "type": "model_selected",
                            "current_model": selected.get("current_model", ""),
                            "current_checkpoint": selected.get(
                                "current_checkpoint", ""
                            ),
                            "current_is_sft": selected.get("current_is_sft", False),
                        }
                    )
                elif action == "start_benchmark":
                    if benchmark_task and not benchmark_task.done():
                        raise ValueError("Benchmark already running")
                    puzzle_count = payload.get("puzzle_count")
                    pc = int(puzzle_count) if puzzle_count is not None else None
                    engine._ensure_loaded()
                    benchmark_task = asyncio.create_task(run_benchmark(pc))
                    response = json.dumps(
                        {"type": "benchmark_started", "puzzle_count": pc}
                    )
                elif action == "stop_benchmark":
                    _benchmark_cancel.set()
                    response = json.dumps({"type": "benchmark_stopping"})
                else:
                    raise ValueError(f"Unsupported action: {action}")
            else:
                raise ValueError("Message payload must be an object")
        except json.JSONDecodeError:
            response = await asyncio.to_thread(engine.infer, message)
        except Exception as exc:
            response = json.dumps({"type": "error", "message": str(exc)})
        await websocket.send(response)


async def main() -> None:
    async with serve(echo, HOST, PORT) as server:
        print(f"WebSocket server running at ws://{HOST}:{PORT}")
        print(f"Training dir: {engine.training_dir}")
        print(f"Checkpoint: {engine.ckpt_path}")
        print(f"Device: {engine.device}")
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
