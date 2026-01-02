#!/usr/bin/env python3
"""
Automated evaluation: pit the trained policy against a UCI engine (e.g., Stockfish).

Example:
  python eval_vs_engine.py --model runs/policy_basic/policy_final.pt --engine stockfish --games 10
"""

from __future__ import annotations

import argparse
import shlex
from collections import Counter
from pathlib import Path
from typing import Tuple

import chess
import chess.engine
import torch

from policy_inference import load_model, choose_move


def parse_engine_command(cmd: str) -> list[str]:
    return shlex.split(cmd)


def run_single_game(
    model,
    engine: chess.engine.SimpleEngine,
    engine_color: chess.Color,
    device: torch.device,
    temperature: float,
    limit: chess.engine.Limit,
    max_moves: int,
    start_fen: str,
) -> Tuple[chess.Board, int, bool]:
    board = chess.Board(start_fen)
    ply = 0

    while not board.is_game_over() and ply < max_moves:
        if board.turn == engine_color:
            result = engine.play(board, limit)
            move = result.move
        else:
            move = choose_move(model, board, device, temperature=temperature)
        board.push(move)
        ply += 1

    hit_move_cap = ply >= max_moves and board.outcome(claim_draw=True) is None
    return board, ply, hit_move_cap


def main():
    ap = argparse.ArgumentParser(description="Evaluate the ChessTransformerPolicy against a UCI engine.")
    ap.add_argument("--model", type=str, required=True, help="Path to policy_final.pt or checkpoint.")
    ap.add_argument("--engine", type=str, default="stockfish", help="UCI engine command (e.g., 'stockfish').")
    ap.add_argument("--games", type=int, default=10, help="Number of games to play.")
    ap.add_argument("--engine-move-time", type=float, default=0.1, help="Engine time per move in seconds.")
    ap.add_argument("--engine-depth", type=int, default=None, help="Optional fixed search depth.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the policy.")
    ap.add_argument("--max-moves", type=int, default=200, help="Max plies before declaring a draw.")
    ap.add_argument("--fen", type=str, default=chess.STARTING_FEN, help="Starting position FEN.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU for the policy model.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg = load_model(Path(args.model), device)
    print(f"[info] loaded model from {args.model} on {device.type}")

    cmd = parse_engine_command(args.engine)
    if not cmd:
        raise ValueError("Engine command is empty.")
    print(f"[info] starting engine: {' '.join(cmd)}")
    engine = chess.engine.SimpleEngine.popen_uci(cmd)

    limit_kwargs = {}
    if args.engine_move_time is not None:
        limit_kwargs["time"] = args.engine_move_time
    if args.engine_depth is not None:
        limit_kwargs["depth"] = args.engine_depth
    if not limit_kwargs:
        raise ValueError("Specify at least --engine-move-time or --engine-depth.")
    limit = chess.engine.Limit(**limit_kwargs)

    stats = Counter()
    total_plies = 0

    try:
        for game_idx in range(args.games):
            engine_color = chess.WHITE if (game_idx % 2 == 0) else chess.BLACK
            board, plies, hit_cap = run_single_game(
                model=model,
                engine=engine,
                engine_color=engine_color,
                device=device,
                temperature=args.temperature,
                limit=limit,
                max_moves=args.max_moves,
                start_fen=args.fen,
            )
            total_plies += plies

            outcome = board.outcome(claim_draw=True)
            result = board.result(claim_draw=True)
            if hit_cap and outcome is None:
                stats["draw"] += 1
                reason = "max_moves"
            elif outcome is None:
                stats["unfinished"] += 1
                reason = "unfinished"
            else:
                winner = outcome.winner
                model_color = chess.WHITE if engine_color == chess.BLACK else chess.BLACK
                if winner is None:
                    stats["draw"] += 1
                elif winner == model_color:
                    stats["model_win"] += 1
                else:
                    stats["engine_win"] += 1
                reason = outcome.termination.name

            print(
                f"[game {game_idx+1}/{args.games}] engine as {'white' if engine_color else 'black'} | "
                f"result {result} | plies {plies} | reason {reason}"
            )
    finally:
        engine.quit()

    print("=== Summary ===")
    total = max(1, args.games)
    for key in ["model_win", "draw", "engine_win", "unfinished"]:
        if stats[key]:
            print(f"{key:12s}: {stats[key]} ({stats[key]/total:.2%})")
    avg_len = total_plies / max(1, args.games)
    print(f"average plies: {avg_len:.1f}")


if __name__ == "__main__":
    main()
