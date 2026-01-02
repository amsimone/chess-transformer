#!/usr/bin/env python3
"""
Quick CLI to test a trained ChessTransformerPolicy by playing against it.

Example:
  python play.py --model runs/policy_basic/policy_final.pt --play-as white
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import chess
import torch

from policy_inference import load_model, choose_move


def print_status(board: chess.Board) -> None:
    print(board)
    print(f"FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")


def play_loop(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg = load_model(Path(args.model), device)
    print(f"[info] loaded model from {args.model} on {device.type}")
    if cfg:
        print(
            "[info] model config: "
            f"d_model={cfg.get('d_model')} | n_layers={cfg.get('n_layers')} | n_heads={cfg.get('n_heads')}"
        )

    board = chess.Board(args.fen)
    user_color = chess.WHITE if args.play_as.lower() == "white" else chess.BLACK
    max_moves = args.max_moves

    move_idx = 1
    print_status(board)

    while not board.is_game_over() and (max_moves is None or move_idx <= max_moves):
        if board.turn == user_color:
            move = prompt_user_move(board)
            if move is None:
                print("[info] stopping the game.")
                return
            board.push(move)
        else:
            move = choose_move(model, board, device, temperature=args.temperature)
            board.push(move)
            player = "White" if board.turn == chess.BLACK else "Black"
            print(f"[engine] {player} played {move.uci()}")

        print_status(board)
        move_idx += 1

    print("[info] game over.")
    print(board.result(claim_draw=True))
    print(board.outcome(claim_draw=True))


def prompt_user_move(board: chess.Board) -> Optional[chess.Move]:
    while True:
        raw = input("Your move (uci or 'quit'): ").strip()
        if raw.lower() in {"quit", "exit"}:
            return None
        try:
            move = board.parse_uci(raw)
        except ValueError:
            print("Invalid UCI string. Example: e2e4")
            continue
        if move not in board.legal_moves:
            print("Illegal move in this position.")
            continue
        return move


def main():
    ap = argparse.ArgumentParser(description="Play against a trained ChessTransformerPolicy.")
    ap.add_argument("--model", type=str, required=True, help="Path to policy_final.pt or checkpoint.")
    ap.add_argument("--fen", type=str, default=chess.STARTING_FEN, help="Starting position FEN.")
    ap.add_argument("--play-as", type=str, default="white", choices=["white", "black"])
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for moves.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    ap.add_argument("--max-moves", type=int, default=None, help="Optional max plies to play.")
    args = ap.parse_args()

    play_loop(args)


if __name__ == "__main__":
    main()
