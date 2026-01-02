# train.py
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterator, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from model import ChessTransformerPolicy

_FILES = "abcdefgh"
_PIECE_CHARS = {
    0: ".",
    1: "P",
    2: "N",
    3: "B",
    4: "R",
    5: "Q",
    6: "K",
    7: "p",
    8: "n",
    9: "b",
    10: "r",
    11: "q",
    12: "k",
}
_PROMO_SUFFIX = {
    0: "",
    1: "=n",
    2: "=b",
    3: "=r",
    4: "=q",
}


def _square_name(idx: int) -> str:
    if idx < 0 or idx >= 64:
        return f"?{idx}?"
    file_idx = idx % 8
    rank = idx // 8 + 1
    return f"{_FILES[file_idx]}{rank}"


def _castling_mask_to_str(mask: int) -> str:
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


def _format_board_ascii(x: torch.Tensor) -> str:
    board = x.view(8, 8)
    rows = []
    for rank_idx in range(7, -1, -1):
        row = " ".join(_PIECE_CHARS.get(int(val), "?") for val in board[rank_idx])
        rows.append(f"{rank_idx + 1} {row}")
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def _move_to_str(y_from: int, y_to: int, y_prom: int) -> str:
    return f"{_square_name(y_from)}{_square_name(y_to)}{_PROMO_SUFFIX.get(y_prom, '')}"


def _describe_sample(batch: Dict[str, torch.Tensor], idx: int = 0) -> str:
    x = batch["X"][idx].cpu()
    stm = int(batch["stm"][idx])
    cr = int(batch["cr"][idx])
    ep = int(batch["ep"][idx])
    y_from = int(batch["y_from"][idx])
    y_to = int(batch["y_to"][idx])
    y_prom = int(batch["y_prom"][idx])
    info = [
        f"sample idx={idx} | stm={'white' if stm == 1 else 'black'} | castling={_castling_mask_to_str(cr)} | ep={_square_name(ep) if ep >= 0 else '-'}",
        f"label move: {_move_to_str(y_from, y_to, y_prom)}",
        _format_board_ascii(x),
    ]
    return "\n".join(info)


def _first_indices(mask: torch.Tensor, limit: int = 5) -> List[int]:
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    return idx[:limit].cpu().tolist()


def validate_batch(batch: Dict[str, torch.Tensor]) -> tuple[bool, List[str]]:
    issues: List[str] = []
    X = batch["X"]
    stm = batch["stm"]
    cr = batch["cr"]
    ep = batch["ep"]
    y_from = batch["y_from"]
    y_to = batch["y_to"]
    y_prom = batch["y_prom"]

    if not torch.all((X >= 0) & (X <= 12)).item():
        bad = torch.nonzero(~((X >= 0) & (X <= 12)), as_tuple=False)
        issues.append(f"piece codes out of range: {bad.size(0)} entries, samples {bad[:5].tolist()}")

    if not torch.all((stm == 0) | (stm == 1)).item():
        issues.append("stm tensor contains values outside {0,1}")

    if not torch.all((cr >= 0) & (cr <= 15)).item():
        issues.append("castling rights mask outside [0,15]")

    if not torch.all(((ep >= 0) & (ep < 64)) | (ep == -1)).item():
        issues.append("en-passant square must be -1 or 0..63")

    if not torch.all((y_from >= 0) & (y_from < 64)).item():
        issues.append("y_from indices outside board range 0..63")

    if not torch.all((y_to >= 0) & (y_to < 64)).item():
        issues.append("y_to indices outside board range 0..63")

    if not torch.all((y_prom >= 0) & (y_prom <= 4)).item():
        issues.append("promotion labels must be in 0..4")

    bsz = X.size(0)
    batch_idx = torch.arange(bsz, device=X.device)
    from_codes = X[batch_idx, y_from]
    empty_mask = from_codes == 0
    if empty_mask.any():
        issues.append(f"{empty_mask.sum().item()} moves originate from empty squares (samples {_first_indices(empty_mask)})")

    white_mask = stm == 1
    white_piece = (from_codes >= 1) & (from_codes <= 6)
    black_piece = (from_codes >= 7) & (from_codes <= 12)
    wrong_color_mask = (white_mask & (~white_piece)) | ((~white_mask) & (~black_piece))
    if wrong_color_mask.any():
        issues.append(
            f"{wrong_color_mask.sum().item()} moves originate from opponent pieces (samples {_first_indices(wrong_color_mask)})"
        )

    return len(issues) == 0, issues


class ShardedNPZDataset(IterableDataset):
    """
    Streams shards like shard_00000.npz. Each shard contains arrays:
      X, stm, cr, ep, y_from, y_to, y_prom
    """

    def __init__(
        self,
        data_dir: Path,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        seed: int = 1234,
        limit_examples: int | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.seed = seed
        self.limit_examples = limit_examples

        self.shards = sorted(self.data_dir.glob("shard_*.npz"))
        if not self.shards:
            raise RuntimeError(f"No shard_*.npz found in {self.data_dir}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = np.random.default_rng(self.seed + (os.getpid() % 10_000))

        shards = list(self.shards)
        if self.shuffle_shards:
            rng.shuffle(shards)

        yielded = 0
        for shard_path in shards:
            with np.load(shard_path) as d:
                n = d["X"].shape[0]
                idx = np.arange(n)
                if self.shuffle_within_shard:
                    rng.shuffle(idx)

                for i in idx:
                    ex = {
                        "X":      d["X"][i].astype(np.int64),     # (64,)
                        "stm":    np.int64(d["stm"][i]),
                        "cr":     np.int64(d["cr"][i]),
                        "ep":     np.int64(d["ep"][i]),
                        "y_from": np.int64(d["y_from"][i]),
                        "y_to":   np.int64(d["y_to"][i]),
                        "y_prom": np.int64(d["y_prom"][i]),
                    }
                    yield ex
                    yielded += 1
                    if self.limit_examples is not None and yielded >= self.limit_examples:
                        return


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    X = torch.tensor(np.stack([b["X"] for b in batch], axis=0), dtype=torch.long)
    stm = torch.tensor([b["stm"] for b in batch], dtype=torch.long)
    cr = torch.tensor([b["cr"] for b in batch], dtype=torch.long)
    ep = torch.tensor([b["ep"] for b in batch], dtype=torch.long)

    y_from = torch.tensor([b["y_from"] for b in batch], dtype=torch.long)
    y_to   = torch.tensor([b["y_to"] for b in batch], dtype=torch.long)
    y_prom = torch.tensor([b["y_prom"] for b in batch], dtype=torch.long)

    return {"X": X, "stm": stm, "cr": cr, "ep": ep, "y_from": y_from, "y_to": y_to, "y_prom": y_prom}


@torch.no_grad()
def estimate_dataset_size(data_dir: Path) -> int:
    total = 0
    for f in sorted(Path(data_dir).glob("shard_*.npz")):
        with np.load(f) as d:
            total += int(d["X"].shape[0])
    return total


def select_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = select_device(force_cpu=args.cpu)
    torch.set_float32_matmul_precision("high")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.estimate_size:
        n = estimate_dataset_size(data_dir)
        print(f"[info] dataset examples/plies: {n}  (approx full moves: {n/2:.1f})")

    ds = ShardedNPZDataset(
        data_dir=data_dir,
        shuffle_shards=True,
        shuffle_within_shard=True,
        seed=args.seed,
        limit_examples=args.limit_examples,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    model = ChessTransformerPolicy(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    # Use torch.amp APIs to avoid deprecation warnings, and pick the backend based on the active device.
    amp_device = device.type if device.type in {"cuda", "mps"} else "cuda"
    amp_enabled = (device.type in {"cuda", "mps"}) and args.amp
    scaler = torch.amp.GradScaler(device=amp_device, enabled=amp_enabled)
    print(f"[info] training on {device.type} | steps={args.total_steps} | amp={'on' if amp_enabled else 'off'}")

    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if amp_enabled and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        print(f"[info] resumed from {resume_path} @ step {start_step}")

    # Simple cosine schedule with warmup
    def lr_mult(step: int):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    ce = nn.CrossEntropyLoss()

    model.train()
    step = start_step
    running = {"loss": 0.0, "from_acc": 0.0, "to_acc": 0.0, "prom_acc": 0.0}
    validated_batches = 0

    pbar = tqdm(
        total=args.total_steps,
        initial=step,
        desc="training",
        unit="step",
        dynamic_ncols=True,
    )
    try:
        for batch in dl:
            if step >= args.total_steps:
                break

            if args.debug_validate_batches and validated_batches < args.debug_validate_batches:
                ok, issues = validate_batch(batch)
                if ok:
                    pbar.write(f"[debug] batch-check #{validated_batches + 1} passed ({batch['X'].size(0)} examples)")
                else:
                    pbar.write(f"[debug] batch-check #{validated_batches + 1} found issues:")
                    for msg in issues:
                        pbar.write(f"    - {msg}")
                validated_batches += 1

            if args.debug_print_sample_every > 0:
                steps_since_resume = step - start_step
                if steps_since_resume % args.debug_print_sample_every == 0:
                    pbar.write("[debug] batch snapshot:\\n" + _describe_sample(batch, idx=0))

            X   = batch["X"].to(device, non_blocking=True)
            stm = batch["stm"].to(device, non_blocking=True)
            cr  = batch["cr"].to(device, non_blocking=True)
            ep  = batch["ep"].to(device, non_blocking=True)

            y_from = batch["y_from"].to(device, non_blocking=True)
            y_to   = batch["y_to"].to(device, non_blocking=True)
            y_prom = batch["y_prom"].to(device, non_blocking=True)

            # LR schedule
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_mult(step)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
                from_logits, to_logits, prom_logits = model(X, stm, cr, ep)
                loss_from = ce(from_logits, y_from)
                loss_to   = ce(to_logits, y_to)
                loss_prom = ce(prom_logits, y_prom)

                # Basic combined loss; you can reweight later
                loss = loss_from + loss_to + 0.2 * loss_prom

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # metrics
            with torch.no_grad():
                from_pred = from_logits.argmax(dim=-1)
                to_pred   = to_logits.argmax(dim=-1)
                prom_pred = prom_logits.argmax(dim=-1)

                from_acc = (from_pred == y_from).float().mean().item()
                to_acc   = (to_pred == y_to).float().mean().item()
                prom_acc = (prom_pred == y_prom).float().mean().item()

            running["loss"] += loss.item()
            running["from_acc"] += from_acc
            running["to_acc"] += to_acc
            running["prom_acc"] += prom_acc

            step += 1
            pbar.update(1)

            if step % args.log_every == 0:
                k = args.log_every
                loss_avg = running["loss"] / k
                from_avg = running["from_acc"] / k
                to_avg   = running["to_acc"] / k
                prom_avg = running["prom_acc"] / k
                lr_val = opt.param_groups[0]["lr"]

                msg = (
                    f"step {step:6d} | "
                    f"loss {loss_avg:.4f} | "
                    f"from_acc {from_avg:.3f} | "
                    f"to_acc {to_avg:.3f} | "
                    f"prom_acc {prom_avg:.3f} | "
                    f"lr {lr_val:.2e}"
                )
                pbar.write(msg)
                pbar.set_postfix(
                    loss=f"{loss_avg:.4f}",
                    from_acc=f"{from_avg:.3f}",
                    to_acc=f"{to_avg:.3f}",
                    prom_acc=f"{prom_avg:.3f}",
                    lr=f"{lr_val:.2e}",
                )
                for key in running:
                    running[key] = 0.0

            if step % args.ckpt_every == 0:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                }
                if amp_enabled:
                    ckpt["scaler"] = scaler.state_dict()
                ckpt_path = out_dir / f"ckpt_step_{step:06d}.pt"
                torch.save(ckpt, ckpt_path)
                pbar.write(f"[info] saved {ckpt_path}")
    finally:
        pbar.close()

    # final save
    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "policy_final.pt")
    print(f"[info] saved {out_dir / 'policy_final.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/ds")
    ap.add_argument("--out_dir", type=str, default="runs/policy_basic")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--total_steps", type=int, default=20_000)

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit_examples", type=int, default=None, help="debug: stop after N examples yielded")
    ap.add_argument("--estimate_size", action="store_true")
    ap.add_argument("--debug_validate_batches", type=int, default=0,
                    help="Run range/consistency checks on the first N batches before training (0=disable)")
    ap.add_argument("--debug_print_sample_every", type=int, default=0,
                    help="Print a decoded sample from the current batch every N steps (0=disable)")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
