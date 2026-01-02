#!/usr/bin/env python3
"""
Fetch + process chess games for training a policy model.

Default source: "Lichess Elite Database" monthly zip files:
  https://database.nikonoel.fr/lichess_elite_YYYY-MM.zip

Outputs: sharded .npz files with arrays:
  X        int8  [N, 64]   piece codes per square (a1..h8)
  stm      uint8 [N]       side to move (1=white, 0=black)
  cr       uint8 [N]       castling rights bitmask (KQkq)
  ep       int16 [N]       en-passant square (0..63) or -1
  y_from   uint8 [N]       from-square (0..63)
  y_to     uint8 [N]       to-square (0..63)
  y_prom   uint8 [N]       promotion (0 none, 1 n, 2 b, 3 r, 4 q)

Usage examples:
  python chess_data_pipeline.py fetch --months 2025-09 2025-10 --out data/raw
  python chess_data_pipeline.py process --raw data/raw --out data/ds --max_games 200000 --positions_per_game 12
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, List

import numpy as np

# pip install python-chess requests tqdm
import chess
import chess.pgn
import requests
from tqdm import tqdm


ELITE_URL_TMPL = "https://database.nikonoel.fr/lichess_elite_{month}.zip"
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


# ---------- Encoding ----------

# piece codes: 0 empty
# 1..6 white: pawn, knight, bishop, rook, queen, king
# 7..12 black: pawn..king
_PIECE_TO_CODE = {
    (chess.PAWN, True): 1,
    (chess.KNIGHT, True): 2,
    (chess.BISHOP, True): 3,
    (chess.ROOK, True): 4,
    (chess.QUEEN, True): 5,
    (chess.KING, True): 6,
    (chess.PAWN, False): 7,
    (chess.KNIGHT, False): 8,
    (chess.BISHOP, False): 9,
    (chess.ROOK, False): 10,
    (chess.QUEEN, False): 11,
    (chess.KING, False): 12,
}

_PROMO_TO_CODE = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

_CODE_TO_PIECE = {code: (piece_type, color) for (piece_type, color), code in _PIECE_TO_CODE.items()}
_CODE_TO_PROMO = {code: piece_type for piece_type, code in _PROMO_TO_CODE.items()}


def _castling_mask_to_fen(mask: int) -> str:
    rights = []
    if mask & 1:
        rights.append("K")
    if mask & 2:
        rights.append("Q")
    if mask & 4:
        rights.append("k")
    if mask & 8:
        rights.append("q")
    return "".join(rights) or "-"


def decode_position(x: np.ndarray, stm: int, cr: int, ep: int) -> chess.Board:
    """Rebuild a chess.Board from encoded arrays for inspection/debugging."""
    board = chess.Board.empty()
    board.turn = chess.WHITE if int(stm) == 1 else chess.BLACK
    board.set_castling_fen(_castling_mask_to_fen(int(cr)))
    board.ep_square = int(ep) if int(ep) >= 0 else None

    for sq, code in enumerate(x.tolist()):
        if code == 0:
            continue
        piece = _CODE_TO_PIECE.get(int(code))
        if piece is None:
            continue
        piece_type, color = piece
        board.set_piece_at(sq, chess.Piece(piece_type, color))

    return board


def inspect_dataset(
    dataset_dir: Path,
    num_samples: int = 5,
    shard_index: Optional[int] = None,
    seed: int = 1234,
) -> None:
    """Print a few decoded samples from stored shards for manual inspection."""

    shards = sorted(dataset_dir.glob("shard_*.npz"))
    if not shards:
        raise RuntimeError(f"No shard_*.npz files found in {dataset_dir}")

    if shard_index is not None:
        if shard_index < 0 or shard_index >= len(shards):
            raise ValueError(f"Shard index {shard_index} out of range (found {len(shards)} shards)")
        shard_choices = [shards[shard_index]]
    else:
        shard_choices = shards

    rng = np.random.default_rng(seed)

    for sample_idx in range(num_samples):
        shard_path = shard_choices[sample_idx % len(shard_choices)] if shard_index is not None else shard_choices[rng.integers(0, len(shard_choices))]
        with np.load(shard_path) as data:
            n = data["X"].shape[0]
            idx = int(rng.integers(0, n))
            x = data["X"][idx]
            stm = int(data["stm"][idx])
            cr = int(data["cr"][idx])
            ep = int(data["ep"][idx])
            y_from = int(data["y_from"][idx])
            y_to = int(data["y_to"][idx])
            y_prom = int(data["y_prom"][idx])

        board = decode_position(x, stm, cr, ep)
        promo_piece = _CODE_TO_PROMO.get(y_prom, None)
        move = chess.Move(y_from, y_to, promotion=promo_piece)

        legal = board.is_legal(move)
        try:
            san = board.san(move)
        except ValueError:
            san = "(illegal move for reconstructed board)"

        print(f"sample {sample_idx + 1}/{num_samples} | shard={shard_path.name} | idx={idx}")
        print(f"  stm={'white' if stm == 1 else 'black'} | castling={_castling_mask_to_fen(cr)} | ep={(chess.square_name(ep) if ep >= 0 else '-')}")
        print(f"  move={move.uci()} | san={san} | legal={legal}")
        print(board)
        print("-" * 40)

def encode_board(board: chess.Board) -> np.ndarray:
    """Return int8[64] with squares in python-chess order (a1=0 .. h8=63)."""
    x = np.zeros(64, dtype=np.int8)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        x[sq] = _PIECE_TO_CODE[(p.piece_type, p.color)]
    return x

def encode_castling_rights(board: chess.Board) -> int:
    """Bitmask KQkq -> 1,2,4,8."""
    m = 0
    if board.has_kingside_castling_rights(chess.WHITE): m |= 1
    if board.has_queenside_castling_rights(chess.WHITE): m |= 2
    if board.has_kingside_castling_rights(chess.BLACK): m |= 4
    if board.has_queenside_castling_rights(chess.BLACK): m |= 8
    return m

def encode_ep_square(board: chess.Board) -> int:
    return int(board.ep_square) if board.ep_square is not None else -1

def encode_move(move: chess.Move) -> Tuple[int, int, int]:
    return int(move.from_square), int(move.to_square), int(_PROMO_TO_CODE.get(move.promotion, 0))


# ---------- Downloading ----------

def _download_with_resume(url: str, dst: Path, timeout_s: int = 30, max_retries: int = 5) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    existing = dst.stat().st_size if dst.exists() else 0
    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout_s, headers=headers) as r:
                r.raise_for_status()

                # If server ignores Range, it may return full content; overwrite in that case.
                mode = "ab" if ("Content-Range" in r.headers and existing > 0) else "wb"
                if mode == "wb":
                    existing = 0

                total = None
                if "Content-Length" in r.headers:
                    total = int(r.headers["Content-Length"]) + existing

                pbar = tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    desc=dst.name,
                    leave=True,
                )

                with open(dst, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))
                pbar.close()

            return
        except Exception as e:
            if attempt == max_retries:
                raise
            sleep_s = min(2 ** attempt, 30)
            print(f"[warn] download failed (attempt {attempt}/{max_retries}): {e} | retrying in {sleep_s}s", file=sys.stderr)
            time.sleep(sleep_s)

def fetch_elite(months: List[str], out_dir: Path) -> List[Path]:
    out_paths = []
    for m in months:
        if not MONTH_RE.match(m):
            raise ValueError(f"Invalid month '{m}'. Use YYYY-MM, e.g. 2025-11")

        url = ELITE_URL_TMPL.format(month=m)
        dst = out_dir / f"lichess_elite_{m}.zip"
        print(f"[info] fetching {url}")
        _download_with_resume(url, dst)
        out_paths.append(dst)
    return out_paths


# ---------- Processing ----------

@dataclass
class ShardWriter:
    out_dir: Path
    shard_size: int
    shard_idx: int = 0
    n_in_shard: int = 0

    X: Optional[np.ndarray] = None
    stm: Optional[np.ndarray] = None
    cr: Optional[np.ndarray] = None
    ep: Optional[np.ndarray] = None
    y_from: Optional[np.ndarray] = None
    y_to: Optional[np.ndarray] = None
    y_prom: Optional[np.ndarray] = None

    def _alloc(self):
        self.X = np.zeros((self.shard_size, 64), dtype=np.int8)
        self.stm = np.zeros((self.shard_size,), dtype=np.uint8)
        self.cr = np.zeros((self.shard_size,), dtype=np.uint8)
        self.ep = np.zeros((self.shard_size,), dtype=np.int16)
        self.y_from = np.zeros((self.shard_size,), dtype=np.uint8)
        self.y_to = np.zeros((self.shard_size,), dtype=np.uint8)
        self.y_prom = np.zeros((self.shard_size,), dtype=np.uint8)

    def add(self, board: chess.Board, move: chess.Move):
        if self.n_in_shard >= self.shard_size:
            self.flush()
        if self.X is None:
            self._alloc()

        i = self.n_in_shard
        self.X[i, :] = encode_board(board)
        self.stm[i] = 1 if board.turn == chess.WHITE else 0
        self.cr[i] = encode_castling_rights(board)
        self.ep[i] = encode_ep_square(board)

        f, t, p = encode_move(move)
        self.y_from[i] = f
        self.y_to[i] = t
        self.y_prom[i] = p

        self.n_in_shard += 1

    def flush(self):
        if self.X is None or self.n_in_shard == 0:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fn = self.out_dir / f"shard_{self.shard_idx:05d}.npz"
        np.savez_compressed(
            fn,
            X=self.X[: self.n_in_shard],
            stm=self.stm[: self.n_in_shard],
            cr=self.cr[: self.n_in_shard],
            ep=self.ep[: self.n_in_shard],
            y_from=self.y_from[: self.n_in_shard],
            y_to=self.y_to[: self.n_in_shard],
            y_prom=self.y_prom[: self.n_in_shard],
        )
        print(f"[info] wrote {fn} with {self.n_in_shard} positions")
        self.shard_idx += 1
        self.n_in_shard = 0
        self.X = self.stm = self.cr = self.ep = self.y_from = self.y_to = self.y_prom = None


def iter_pgn_games_from_zip(zip_path: Path) -> Iterator[chess.pgn.Game]:
    """Yield games from all .pgn files inside a zip."""
    with zipfile.ZipFile(zip_path, "r") as z:
        pgn_names = [n for n in z.namelist() if n.lower().endswith(".pgn")]
        if not pgn_names:
            raise RuntimeError(f"No .pgn found inside {zip_path}")

        for name in pgn_names:
            with z.open(name, "r") as fbin:
                # python-chess expects a text stream
                text = io.TextIOWrapper(fbin, encoding="utf-8", errors="replace", newline="")
                while True:
                    game = chess.pgn.read_game(text)
                    if game is None:
                        break
                    yield game


def _get_int(headers: chess.pgn.Headers, key: str, default: int = 0) -> int:
    """Return header value as int, defaulting for missing or '?' entries."""
    try:
        value = headers.get(key)
        if value is None or value == "?":
            return default
        return int(value)
    except Exception:
        return default


def process_zips(
    zip_files: List[Path],
    out_dir: Path,
    shard_size: int = 200_000,
    max_games: Optional[int] = None,
    positions_per_game: int = 0,
    min_elo: Optional[int] = None,
    max_elo: Optional[int] = None,
) -> None:
    """
    positions_per_game:
      0 = take all plies
      N > 0 = sample up to N plies per game (uniformly across the game)
    """
    writer = ShardWriter(out_dir=out_dir, shard_size=shard_size)

    total_games_seen = 0
    total_games_kept = 0
    total_plies_kept = 0
    total_examples_written = 0

    for zp in zip_files:
        print(f"[info] processing {zp}")
        for game in iter_pgn_games_from_zip(zp):
            board = game.board()
            moves = list(game.mainline_moves())
            total_games_seen += 1

            if not moves:
                continue

            headers = game.headers
            white_elo = _get_int(headers, "WhiteElo", 0)
            black_elo = _get_int(headers, "BlackElo", 0)

            if min_elo is not None and (white_elo < min_elo or black_elo < min_elo):
                continue
            if max_elo is not None and (white_elo > max_elo or black_elo > max_elo):
                continue

            total_games_kept += 1

            if positions_per_game and positions_per_game > 0 and len(moves) > positions_per_game:
                # sample indices without replacement, keep chronological order
                idx = np.sort(np.random.choice(len(moves), size=positions_per_game, replace=False))
                chosen = set(int(i) for i in idx)
            else:
                chosen = None

            for ply, mv in enumerate(moves):
                if chosen is None or ply in chosen:
                    writer.add(board, mv)
                    total_plies_kept += 1
                    total_examples_written += 1
                board.push(mv)

            if max_games is not None and total_games_kept >= max_games:
                writer.flush()
                print(f"[info] reached max_games={max_games}, stopping")
                print(f"games_seen: {total_games_seen}")
                print(f"games_kept: {total_games_kept}")
                print(f"plies_kept: {total_plies_kept}")
                print(f"examples_written: {total_examples_written}")
                if total_plies_kept:
                    print(f"approx fullmoves: {total_plies_kept / 2}")
                return

    writer.flush()
    print(f"games_seen: {total_games_seen}")
    print(f"games_kept: {total_games_kept}")
    print(f"plies_kept: {total_plies_kept}")
    print(f"examples_written: {total_examples_written}")
    if total_plies_kept:
        print(f"approx fullmoves: {total_plies_kept / 2}")


def summarize_dataset(dataset_dir: Path) -> None:
    shards = sorted(dataset_dir.glob("shard_*.npz"))
    if not shards:
        raise RuntimeError(f"No shard_*.npz files found in {dataset_dir}")

    n_examples = 0
    for shard in shards:
        with np.load(shard) as data:
            if "X" not in data:
                continue
            n_examples += int(data["X"].shape[0])

    print(f"examples / plies: {n_examples}")
    if n_examples:
        print(f"approx fullmoves: {n_examples / 2}")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_fetch = sub.add_parser("fetch", help="Download monthly elite zip files")
    ap_fetch.add_argument("--months", nargs="+", required=True, help="e.g. 2025-10 2025-11")
    ap_fetch.add_argument("--out", type=Path, default=Path("data/raw"))

    ap_proc = sub.add_parser("process", help="Convert downloaded zips into model-ready shards")
    ap_proc.add_argument("--raw", type=Path, default=Path("data/raw"), help="directory containing downloaded .zip files")
    ap_proc.add_argument("--out", type=Path, default=Path("data/dataset"))
    ap_proc.add_argument("--shard_size", type=int, default=200_000)
    ap_proc.add_argument("--max_games", type=int, default=None)
    ap_proc.add_argument("--positions_per_game", type=int, default=0,
                         help="0=all plies, else sample up to N plies per game")
    ap_proc.add_argument("--min_elo", type=int, default=None,
                         help="Require both players to have Elo >= this value")
    ap_proc.add_argument("--max_elo", type=int, default=None,
                         help="Require both players to have Elo <= this value")

    ap_stats = sub.add_parser("stats", help="Summarize stored dataset shards")
    ap_stats.add_argument("--dataset", type=Path, default=Path("data/dataset"),
                          help="Directory containing shard_*.npz files")

    ap_inspect = sub.add_parser("inspect", help="Decode a few random samples from stored shards")
    ap_inspect.add_argument("--dataset", type=Path, default=Path("data/dataset"),
                            help="Directory containing shard_*.npz files")
    ap_inspect.add_argument("--num_samples", type=int, default=5, help="How many positions to display")
    ap_inspect.add_argument("--shard_index", type=int, default=None,
                            help="If provided, only sample from this shard index (0-based)")
    ap_inspect.add_argument("--seed", type=int, default=1234, help="RNG seed for sampling examples")

    args = ap.parse_args()

    if args.cmd == "fetch":
        fetch_elite(args.months, args.out)

    elif args.cmd == "process":
        zip_files = sorted(args.raw.glob("*.zip"))
        if not zip_files:
            raise RuntimeError(f"No .zip files found in {args.raw}. Did you run fetch?")
        if args.min_elo is not None and args.max_elo is not None and args.min_elo > args.max_elo:
            raise ValueError(f"min_elo ({args.min_elo}) cannot exceed max_elo ({args.max_elo})")
        process_zips(
            zip_files=zip_files,
            out_dir=args.out,
            shard_size=args.shard_size,
            max_games=args.max_games,
            positions_per_game=args.positions_per_game,
            min_elo=args.min_elo,
            max_elo=args.max_elo,
        )

    elif args.cmd == "stats":
        summarize_dataset(args.dataset)

    elif args.cmd == "inspect":
        inspect_dataset(args.dataset, num_samples=args.num_samples, shard_index=args.shard_index, seed=args.seed)

if __name__ == "__main__":
    main()
