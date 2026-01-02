# model.py
from __future__ import annotations

import math
import torch
import torch.nn as nn


class ChessTransformerPolicy(nn.Module):
    """
    Input:
      X   (B, 64) int64: piece codes per square (0..12), a1..h8 (python-chess order)
      stm (B,)    int64: side to move (1=white, 0=black)
      cr  (B,)    int64: castling rights bitmask (0..15) for KQkq
      ep  (B,)    int64: en-passant square (0..63) or -1

    Output logits:
      from_logits (B, 64)
      to_logits   (B, 64)
      prom_logits (B, 5)   (0 none, 1 n, 2 b, 3 r, 4 q)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.piece_emb = nn.Embedding(13, d_model)  # 0..12
        self.square_emb = nn.Embedding(64, d_model)

        # Global features as embeddings, added to every token
        self.stm_emb = nn.Embedding(2, d_model)     # 0/1
        self.cr_emb = nn.Embedding(16, d_model)     # 0..15
        self.ep_emb = nn.Embedding(65, d_model)     # 0..63, plus 64 = "none"

        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # norm_first=True forces nested tensors off; disable warning.
        )

        self.norm = nn.LayerNorm(d_model)

        # Heads: predict from/to from per-square representations; promotion from pooled representation
        self.from_head = nn.Linear(d_model, 1)  # applied per token -> (B,64,1) -> (B,64)
        self.to_head   = nn.Linear(d_model, 1)

        self.prom_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 5),
        )

        # init
        self._reset_parameters()

    def _reset_parameters(self):
        # Transformer layers handle their init; keep embeddings reasonably scaled
        nn.init.normal_(self.piece_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.square_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.stm_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cr_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.ep_emb.weight, mean=0.0, std=0.02)

    def forward(self, X, stm, cr, ep):
        """
        X:   (B,64) long
        stm: (B,)   long
        cr:  (B,)   long
        ep:  (B,)   long, -1 means none
        """
        B = X.size(0)
        device = X.device

        squares = torch.arange(64, device=device).unsqueeze(0).expand(B, 64)

        # Base token embeddings (board)
        tok = self.piece_emb(X) + self.square_emb(squares)

        # Global conditioning (broadcast to all tokens)
        ep_idx = torch.where(ep >= 0, ep, torch.full_like(ep, 64))
        g = self.stm_emb(stm) + self.cr_emb(cr) + self.ep_emb(ep_idx)
        tok = tok + g.unsqueeze(1)

        tok = self.drop(tok)

        h = self.encoder(tok)         # (B,64,d_model)
        h = self.norm(h)

        from_logits = self.from_head(h).squeeze(-1)  # (B,64)
        to_logits   = self.to_head(h).squeeze(-1)    # (B,64)

        pooled = h.mean(dim=1)        # (B,d_model)
        prom_logits = self.prom_head(pooled)  # (B,5)

        return from_logits, to_logits, prom_logits
