import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

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
        self._run_once = None

    def _ensure_loaded(self) -> None:
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
            candidates = [
                self.training_dir / "storage/out/small_llm_wordpiece/best.pt",
                self.training_dir / "model/best.pt",
            ]
            self.ckpt_path = next((c for c in candidates if c.exists()), candidates[0])

        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                "Checkpoint not found. Set checkpoint_path in api/server.config.json, "
                "set CHESSINGTON_CKPT, or place model at "
                f"{self.ckpt_path}"
            )

        ckpt = load_checkpoint(self.ckpt_path, self.device)
        self.model, self.tokenizer, self.meta, _ = build_model_from_checkpoint(
            ckpt, self.device
        )

    def infer(self, prompt: str) -> str:
        self._ensure_loaded()
        if self._run_once is None:
            raise RuntimeError("Inference function not initialized")
        output = self._run_once(
            self.model,
            self.tokenizer,
            self.meta,
            prompt,
            self.max_new_tokens,
            self.temperature,
            self.top_k,
            self.device,
        )
        return output


engine = InferenceEngine()
HOST = str(os.getenv("CHESSINGTON_HOST") or CONFIG.get("host", "localhost"))
PORT = int(os.getenv("CHESSINGTON_PORT") or CONFIG.get("port", 8765))


async def echo(websocket):
    async for message in websocket:
        try:
            response = await asyncio.to_thread(engine.infer, message)
        except Exception as exc:
            response = f"[inference_error] {exc}"
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
