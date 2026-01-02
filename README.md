## Chess Transformer Data Pipeline

`chess_data_pipeline.py` downloads monthly batches of Lichess Elite PGNs and converts them into sharded NumPy datasets for training. The script now exposes four subcommands: `fetch`, `process`, `stats`, and `inspect`.

### Prerequisites

- Python 3.9+
- `pip install python-chess requests tqdm numpy`

All commands below assume you run them from the repository root.

### 1. Fetch monthly PGN archives

```bash
python chess_data_pipeline.py fetch \
  --months 2024-09 2024-10 2024-11 2024-12 2025-01 2025-02 \
  --out data/raw
```

This downloads `lichess_elite_<month>.zip` files to `data/raw`. Fetching 5–6 recent months typically yields ~600k+ high-quality games, which is enough to produce ≈10M encoded positions once processed with the defaults below. You can resume interrupted downloads.

### 2. Process PGNs into training shards

```bash
python chess_data_pipeline.py process \
  --raw data/raw \
  --out data/dataset \
  --shard_size 200000 \
  --positions_per_game 16 \
  --min_elo 2000 \
  --max_elo 2800 \
  --max_games 600000
```

Key flags:

- `--positions_per_game 0` keeps every ply; set it >0 to uniformly sample plies per game.
- `--min_elo` / `--max_elo` require both players’ ratings to fall within a range.
- `--max_games` stops after keeping the specified number of games (post-filter).
- `--shard_size` controls how many positions land in each `shard_<idx>.npz`.

### How much data do I need?

The default training loop (20k steps × batch size 256) consumes roughly **5.1 million** positions per pass through the dataset. For stable convergence you want at least 2–3 passes without heavy repetition, so target **10–15 million** examples (≈50–75 shards at the 200k shard size). Rules of thumb:

- `examples_per_epoch = total_steps × batch_size`. Adjust `total_steps` or `batch_size` if you have less data.
- Shard count × shard size = total positions. With 200k shards, 50 shards ≈10M examples.
- The default processing command above (600k games × 16 positions per game) yields ~9.6M samples before filtering; increase `--max_games` or fetch more months if you need extra coverage.
- While training, pass `--estimate_size` to `train.py` to print the current dataset size and confirm you meet the target.

During processing the script prints counters for `games_seen`, `games_kept`, `plies_kept`, `examples_written`, and an approximate number of full moves.

### 3. Inspect stored shards

To quickly summarize how many training examples currently live in a dataset directory:

```bash
python chess_data_pipeline.py stats --dataset data/dataset
```

The command totals `X` rows across all shards and reports plies plus approximate full moves.

### 4. Decode and spot-check positions

Use the new `inspect` subcommand to sample encoded positions, rebuild them as `python-chess` boards, and print metadata plus the target move. This is useful when sanity-checking preprocessing before a long training run:

```bash
python chess_data_pipeline.py inspect \
  --dataset data/dataset \
  --num_samples 3 \
  --seed 7
```

Add `--shard_index 12` to pull all samples from a specific shard (0-based).

Each sample prints the side to move, castling rights, en-passant square, the encoded move in both UCI and SAN notation, and an ASCII board dump so you can visually confirm the encoding/labels.

## Training + Debug Instrumentation

`train.py` trains the transformer policy from the sharded dataset. In addition to the existing hyper-parameters, two debug-oriented flags were added to help verify input quality during early experiments:

- `--debug_validate_batches N` runs strict range/consistency checks on the first `N` batches yielded by the data loader (e.g., valid piece codes, en-passant squares, and whether labels originate from a legal piece). Any violations are printed with sample indices so you can trace back the underlying shard.
- `--debug_print_sample_every K` prints a decoded snapshot (board ASCII + labeled move) from the current batch every `K` steps. Set `K=1` for an in-depth inspection run, or a larger interval to occasionally spot-check.

Example: run a quick dry-run on CPU that validates two batches and prints a snapshot before starting a longer job:

```bash
python train.py \
  --data_dir data/dataset \
  --cpu \
  --batch_size 32 \
  --limit_examples 512 \
  --total_steps 200 \
  --debug_validate_batches 2 \
  --debug_print_sample_every 50
```

The debug hooks run before tensors move to the accelerator, so they have minimal impact when disabled (default).

### Recommended full training run

Once the dataset checks out, switch to the full configuration designed around ~10M training positions (≈50 shards of 200k each). On a single 24–48GB GPU this setup fits comfortably with AMP enabled:

```bash
python train.py \
  --data_dir data/dataset \
  --out_dir runs/policy_full \
  --batch_size 256 \
  --total_steps 20000 \
  --n_layers 8 \
  --d_model 384 \
  --n_heads 12 \
  --d_ff 2048 \
  --dropout 0.1 \
  --lr 3e-4 \
  --warmup_steps 1000 \
  --ckpt_every 2000 \
  --log_every 100 \
  --amp
```

This consumes ~5.1M samples per pass; with 10M stored examples the model sees roughly two unique passes before repetition. Adjust `--total_steps` or `--batch_size` if your dataset or hardware budget differs, but keep the examples-per-pass close to the total data volume for best coverage.

### Lightweight ways to confirm the model is learning

1. **Hold-out shards for validation.** Copy a few processed shards (e.g., `shard_0064.npz` onward) to `data/val_ds` and never feed them to the training iterator. Periodically run `train.py` with `--limit_examples` against the validation directory inside a small script to log cross-entropy/accuracy without touching checkpoints.
2. **Inspect predicted moves on fixed probes.** Reuse the batch snapshot output or `policy_inference.py` to run the current checkpoint on a curated list of tactical positions. Track whether the model’s top-1 move matches the ground truth over time.
3. **Track move-rank metrics.** For a handful of validation batches, compute the rank of the ground-truth move within the model’s logits (top-1/top-5 accuracy). The trend is far more informative in early training than playing full games versus an engine.
4. **Self-play smoke test (optional).** After the loss/accuracy curves flatten, let the policy play short self-play games via `play.py --engine policy` with a high sampling temperature. You should see legal, sensible openings emerge long before the model can compete with Stockfish.

### Reproducing dataset counts manually

If you prefer a one-off script, the following snippet matches the built-in `stats` command:

```python
from pathlib import Path
import numpy as np

ds = Path("data/dataset")
n_examples = 0
for shard in sorted(ds.glob("shard_*.npz")):
    with np.load(shard) as data:
        n_examples += data["X"].shape[0]
print("examples / plies:", n_examples)
print("approx fullmoves:", n_examples / 2)
```
