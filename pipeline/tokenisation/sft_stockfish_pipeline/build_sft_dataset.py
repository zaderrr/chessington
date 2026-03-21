#!/usr/bin/env python3
"""Build PGN-native SFT examples with Stockfish quality filtering."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.pgn


@dataclass
class PipelineConfig:
    pgn_path: Path
    output_path: Path
    stockfish_path: Path
    nodes: int
    multipv: int
    delta_threshold: int
    win_cp_threshold: int
    loss_cp_threshold: int
    min_elo: int
    max_games: int
    skip_opening_plies: int


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Build Stockfish-filtered SFT JSONL")
    parser.add_argument("--pgn", required=True, help="Input PGN file path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--stockfish", required=True, help="Stockfish binary path")
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--multipv", type=int, default=1)
    parser.add_argument("--delta-threshold", type=int, default=40)
    parser.add_argument("--win-cp-threshold", type=int, default=80)
    parser.add_argument("--loss-cp-threshold", type=int, default=-80)
    parser.add_argument("--min-elo", type=int, default=0)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--skip-opening-plies", type=int, default=8)
    args = parser.parse_args()
    return PipelineConfig(
        pgn_path=Path(args.pgn),
        output_path=Path(args.output),
        stockfish_path=Path(args.stockfish),
        nodes=args.nodes,
        multipv=args.multipv,
        delta_threshold=args.delta_threshold,
        win_cp_threshold=args.win_cp_threshold,
        loss_cp_threshold=args.loss_cp_threshold,
        min_elo=args.min_elo,
        max_games=args.max_games,
        skip_opening_plies=args.skip_opening_plies,
    )


def header_elo(game: chess.pgn.Game, side: str) -> int:
    value = game.headers.get(side, "")
    try:
        return int(value)
    except ValueError:
        return 0


def game_passes_elo_filter(game: chess.pgn.Game, min_elo: int) -> bool:
    if min_elo <= 0:
        return True
    return (
        header_elo(game, "WhiteElo") >= min_elo
        and header_elo(game, "BlackElo") >= min_elo
    )


def append_history_token(
    board: chess.Board, move: chess.Move, history: list[str]
) -> None:
    san = board.san(move)
    if board.turn == chess.WHITE:
        history.append(f"{board.fullmove_number}.{san}")
    else:
        history.append(san)


def unwrap_info(info: object) -> Optional[dict]:
    if isinstance(info, dict):
        return info
    if isinstance(info, list) and info and isinstance(info[0], dict):
        return info[0]
    return None


def cp_from_info(info: object, side_to_move: chess.Color) -> Optional[int]:
    info_dict = unwrap_info(info)
    if info_dict is None:
        return None
    score_obj = info_dict.get("score")
    if score_obj is None:
        return None
    return score_obj.pov(side_to_move).score(mate_score=100000)


def control_token(cp: int, win_cp_threshold: int, loss_cp_threshold: int) -> str:
    if cp >= win_cp_threshold:
        return "<STM_WIN>"
    if cp <= loss_cp_threshold:
        return "<STM_LOSS>"
    return "<STM_DRAW>"


def run(config: PipelineConfig) -> None:
    if not config.pgn_path.exists():
        raise FileNotFoundError(f"PGN not found: {config.pgn_path}")
    if not config.stockfish_path.exists():
        raise FileNotFoundError(f"Stockfish not found: {config.stockfish_path}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    rejected = 0
    processed_games = 0

    pgn_file = config.pgn_path.open("r", encoding="utf-8", errors="replace")
    out_file = config.output_path.open("w", encoding="utf-8")
    engine = chess.engine.SimpleEngine.popen_uci(str(config.stockfish_path))
    try:
        try:
            engine.configure({"UCI_ShowWDL": True})
        except chess.engine.EngineError:
            pass

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            if config.max_games > 0 and processed_games >= config.max_games:
                break

            processed_games += 1
            if not game_passes_elo_filter(game, config.min_elo):
                continue

            board = game.board()
            history: list[str] = []

            for ply, move in enumerate(game.mainline_moves(), start=1):
                side = board.turn
                san_move = board.san(move)

                if ply > config.skip_opening_plies:
                    prefix = " ".join(history).strip()
                    info_best = engine.analyse(
                        board,
                        chess.engine.Limit(nodes=config.nodes),
                        multipv=config.multipv,
                    )
                    best_info = unwrap_info(info_best)
                    if best_info is not None:
                        best_eval_cp = cp_from_info(best_info, side)
                        pv = best_info.get("pv", [])
                        best_move = pv[0] if pv else None

                        board.push(move)
                        info_after = engine.analyse(
                            board,
                            chess.engine.Limit(nodes=config.nodes),
                            multipv=1,
                        )
                        opp_eval_cp = cp_from_info(info_after, board.turn)
                        board.pop()

                        if best_eval_cp is not None and opp_eval_cp is not None:
                            human_eval_cp = -opp_eval_cp
                            delta_cp = best_eval_cp - human_eval_cp
                            if delta_cp <= config.delta_threshold:
                                token = control_token(
                                    best_eval_cp,
                                    config.win_cp_threshold,
                                    config.loss_cp_threshold,
                                )
                                row = {
                                    "prompt": f"{token} {prefix}".strip(),
                                    "completion": san_move,
                                    "control_token": token,
                                    "best_move_san": board.san(best_move)
                                    if best_move
                                    else None,
                                    "human_move_san": san_move,
                                    "eval_cp": best_eval_cp,
                                    "delta_cp": delta_cp,
                                    "ply": ply,
                                    "game_index": processed_games,
                                    "headers": {
                                        "Event": game.headers.get("Event", ""),
                                        "Site": game.headers.get("Site", ""),
                                        "Date": game.headers.get("Date", ""),
                                        "White": game.headers.get("White", ""),
                                        "Black": game.headers.get("Black", ""),
                                        "Result": game.headers.get("Result", ""),
                                        "WhiteElo": game.headers.get("WhiteElo", ""),
                                        "BlackElo": game.headers.get("BlackElo", ""),
                                    },
                                }
                                out_file.write(
                                    json.dumps(row, ensure_ascii=True) + "\n"
                                )
                                kept += 1
                            else:
                                rejected += 1

                append_history_token(board, move, history)
                board.push(move)

            if processed_games % 100 == 0:
                print(
                    f"Processed games={processed_games} kept={kept} rejected={rejected}"
                )

    finally:
        engine.quit()
        out_file.close()
        pgn_file.close()

    print(
        "Done. "
        f"games={processed_games} kept={kept} rejected={rejected} "
        f"output={config.output_path}"
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
