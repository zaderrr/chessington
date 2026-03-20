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
        self.training_dir = (
            Path(str(training_dir)).expanduser() if training_dir else None
        )

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
                infer_small_llm = importlib.import_module("infer_small_llm")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to import inference modules from {self.training_dir}: {exc}"
                ) from exc

            load_checkpoint = infer_small_llm.load_checkpoint
            build_model_from_checkpoint = infer_small_llm.build_model_from_checkpoint
            self._run_once = infer_small_llm.run_once

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


async def echo(websocket):
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
