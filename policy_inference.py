from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import chess

from chess_data_pipeline import (
    _PROMO_TO_CODE,
    encode_board,
    encode_castling_rights,
    encode_ep_square,
)
from model import ChessTransformerPolicy


def load_model(model_path: Path, device: torch.device) -> Tuple[ChessTransformerPolicy, dict]:
    ckpt = torch.load(model_path, map_location=device)
    if "model" in ckpt:
        state = ckpt["model"]
        cfg = ckpt.get("args", {})
    else:
        state = ckpt
        cfg = {}

    model = ChessTransformerPolicy(
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 6),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 1024),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def encode_position(board: chess.Board, device: torch.device) -> dict[str, torch.Tensor]:
    X = torch.tensor(encode_board(board), dtype=torch.long, device=device).unsqueeze(0)
    stm = torch.tensor([1 if board.turn == chess.WHITE else 0], dtype=torch.long, device=device)
    cr = torch.tensor([encode_castling_rights(board)], dtype=torch.long, device=device)
    ep = torch.tensor([encode_ep_square(board)], dtype=torch.long, device=device)
    return {"X": X, "stm": stm, "cr": cr, "ep": ep}


def choose_move(
    model: ChessTransformerPolicy,
    board: chess.Board,
    device: torch.device,
    temperature: float = 0.0,
) -> chess.Move:
    inputs = encode_position(board, device)
    with torch.no_grad():
        from_logits, to_logits, prom_logits = model(
            inputs["X"], inputs["stm"], inputs["cr"], inputs["ep"]
        )

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("No legal moves available.")

    scores = []
    for mv in legal_moves:
        promo_code = _PROMO_TO_CODE.get(mv.promotion, 0)
        logit = (
            from_logits[0, mv.from_square]
            + to_logits[0, mv.to_square]
            + prom_logits[0, promo_code]
        )
        scores.append((logit.item(), mv))

    logits = torch.tensor([s for s, _ in scores], dtype=torch.float32)
    if temperature > 1e-6:
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=0).cpu().numpy()
        idx = int(np.random.choice(len(scores), p=probs))
        return scores[idx][1]
    idx = int(torch.argmax(logits).item())
    return scores[idx][1]
