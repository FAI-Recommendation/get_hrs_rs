"""
Download all runs from wandb project and save history + summary to report/v2_wandb/
Usage: python scripts/download_wandb_runs.py
"""

import os
import json
import pandas as pd
import wandb
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WANDB_ENTITY  = os.environ["WANDB_ENTITY"]
WANDB_PROJECT = os.environ["WANDB_PROJECT"]
OUT_DIR       = Path(__file__).parent.parent / "report" / "v2_wandb"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reorder columns ───────────────────────────────────────────────────────────
def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    KS = [1, 5, 10, 20, 50]
    METRICS = ["recall", "precision", "ndcg", "map", "mrr", "hit_ratio"]

    base_cols   = [c for c in ["_step", "_runtime", "_timestamp", "epoch"] if c in df.columns]
    train_cols  = [c for c in ["train/loss", "train/mf_loss", "train/reg_loss"] if c in df.columns]
    test_cols   = [f"test/{m}@{k}" for m in METRICS for k in KS if f"test/{m}@{k}" in df.columns]
    rest        = [c for c in df.columns if c not in base_cols + train_cols + test_cols]

    return df[base_cols + train_cols + test_cols + rest]


# ── Connect wandb API ──────────────────────────────────────────────────────────
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
api = wandb.Api()

runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
print(f"Found {len(runs)} runs in {WANDB_ENTITY}/{WANDB_PROJECT}\n")

summary_rows = []

for run in runs:
    run_name = run.name
    run_id   = run.id
    state    = run.state
    print(f"  Downloading: {run_name}  [{state}]")

    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── History (metrics theo từng epoch) ─────────────────────────────────────
    history_df = run.history(samples=10_000)  # lấy tối đa 10k steps
    history_df = reorder_columns(history_df)
    history_df.to_csv(run_dir / "history.csv", index=False)

    # ── Config ────────────────────────────────────────────────────────────────
    config = dict(run.config)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ── Summary (best / final values) ─────────────────────────────────────────
    summary = dict(run.summary._json_dict)
    summary["run_name"] = run_name
    summary["run_id"]   = run_id
    summary["state"]    = state
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_rows.append(summary)
    print(f"    Saved → {run_dir.relative_to(OUT_DIR.parent.parent)}")

# ── Tổng hợp summary tất cả run vào 1 CSV ─────────────────────────────────────
if summary_rows:
    df_all = pd.DataFrame(summary_rows)
    # Đưa run_name lên đầu
    cols = ["run_name", "run_id", "state"] + [
        c for c in df_all.columns if c not in ("run_name", "run_id", "state")
    ]
    df_all = df_all[[c for c in cols if c in df_all.columns]]
    out_csv = OUT_DIR / "all_runs_summary.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nAll summaries → {out_csv}")

print("\nDone!")
