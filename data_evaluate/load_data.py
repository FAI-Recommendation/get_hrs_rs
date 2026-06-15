"""
load_data.py — Auto-scan wandb summary.json files → structured results dict.

Usage:
    from load_data import load_all_runs, load_from_csv
    results = load_all_runs("data_wandb/")         # from individual folders
    results = load_from_csv("data_wandb/all_runs_summary.csv")  # from CSV
"""

import json
import os
import re
from pathlib import Path

import pandas as pd

METRICS = ["recall", "precision", "ndcg", "hit_ratio", "map", "mrr"]
K_VALUES = [1, 5, 10, 20]

# Shared figure counter — incremented by _show_and_save in plot_tier1/tier2
_fig_counter = [0]

# Known encoder suffixes in folder names
_ENCODERS = {"clip", "mbnv2"}


def _parse_folder_name(folder_name: str, config: dict) -> dict:
    """Extract model, encoder, sim_type from folder name + config."""
    sim_type = config.get("sim_type", "unknown")
    if sim_type == "tfidf":
        sim_type = "text_only"
    model = config.get("model", None)

    # Encoder: last token after splitting by '_' that matches known encoders
    parts = folder_name.split("_")
    encoder = "unknown"
    for p in reversed(parts):
        if p in _ENCODERS:
            encoder = p
            break

    # Model: if not in config, try to infer from folder name prefix
    if model is None:
        known_models = {"combigcn", "bm3", "freedom", "lightgcn"}
        first = parts[0].lower()
        if first in known_models:
            model = first
        else:
            model = "combigcn"  # default for legacy runs

    return {"model": model, "encoder": encoder, "sim_type": sim_type}


def _extract_metrics(summary: dict) -> dict:
    """Extract test metrics from summary.json into {metric: [val@1, val@5, val@10, val@20]}."""
    out = {}
    for metric in METRICS:
        vals = []
        for k in K_VALUES:
            key = f"test/{metric}@{k}"
            vals.append(summary.get(key, 0.0))
        out[metric] = vals
    return out


def load_all_runs(data_dir: str = "data_wandb") -> pd.DataFrame:
    """
    Scan all subdirectories in data_dir for summary.json + config.json.

    Returns DataFrame with columns:
        run_name, model, encoder, sim_type, best_epoch, epoch,
        recall@1, recall@5, ..., mrr@20
    """
    data_path = Path(data_dir)
    rows = []

    for sub in sorted(data_path.iterdir()):
        if not sub.is_dir():
            continue

        summary_path = sub / "summary.json"
        config_path = sub / "config.json"

        if not summary_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        info = _parse_folder_name(sub.name, config)
        metrics = _extract_metrics(summary)

        row = {
            "run_name": summary.get("run_name", sub.name),
            "model": info["model"],
            "encoder": info["encoder"],
            "sim_type": info["sim_type"],
            "best_epoch": summary.get("best_epoch", None),
            "epoch": summary.get("epoch", None),
            "train_loss": summary.get("train/loss", None),
        }
        for metric in METRICS:
            for i, k in enumerate(K_VALUES):
                row[f"{metric}@{k}"] = metrics[metric][i]

        rows.append(row)

    return pd.DataFrame(rows)


def _parse_run_name(run_name: str) -> dict:
    """Parse model, sim_type, encoder from run_name like 'bm3_multimodal_layers4_..._clip'."""
    known_models = {"combigcn", "bm3", "freedom", "lightgcn"}
    known_sim_types = {"none", "img_only", "tfidf", "multimodal", "multimodal_attention"}

    parts = run_name.split("_")

    model = parts[0] if parts[0] in known_models else "unknown"

    encoder = "unknown"
    for p in reversed(parts):
        if p in _ENCODERS:
            encoder = p
            break

    sim_type = "unknown"
    rest = "_".join(parts[1:])  # skip model prefix
    for st in sorted(known_sim_types, key=len, reverse=True):
        if rest.startswith(st + "_") or rest == st:
            sim_type = st
            break

    if sim_type == "tfidf":
        sim_type = "text_only"

    return {"model": model, "encoder": encoder, "sim_type": sim_type}


def load_from_csv(csv_path: str = "data_wandb/all_runs_summary.csv") -> pd.DataFrame:
    """
    Load from all_runs_summary.csv (exported from wandb).
    Parses model/encoder/sim_type from run_name column.
    """
    raw = pd.read_csv(csv_path)
    rows = []
    for _, r in raw.iterrows():
        info = _parse_run_name(r["run_name"])
        row = {
            "run_name": r["run_name"],
            "model": info["model"],
            "encoder": info["encoder"],
            "sim_type": info["sim_type"],
            "best_epoch": r.get("best_epoch", None),
            "epoch": r.get("epoch", None),
            "train_loss": r.get("train/loss", None),
        }
        for metric in METRICS:
            for k in K_VALUES:
                col = f"test/{metric}@{k}"
                row[f"{metric}@{k}"] = r.get(col, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def to_results_dict(df: pd.DataFrame, label_col: str = None) -> dict:
    """
    Convert DataFrame → results_dict format for plotting.

    results_dict = {
        "label": {
            "recall": [val@1, val@5, val@10, val@20],
            "ndcg": [...],
            ...
        }
    }

    label_col: column to use as dict key. If None, uses
               "{model}_{encoder}({sim_type})" format.
    """
    results = {}
    for _, row in df.iterrows():
        if label_col and label_col in row:
            label = row[label_col]
        else:
            label = f"{row['model']}_{row['encoder']}({row['sim_type']})"

        entry = {}
        for metric in METRICS:
            entry[metric] = [row[f"{metric}@{k}"] for k in K_VALUES]
        results[label] = entry

    return results


def get_best_per_model(df: pd.DataFrame, rank_metric: str = "ndcg@10") -> pd.DataFrame:
    """Return best config per model, ranked by rank_metric."""
    idx = df.groupby("model")[rank_metric].idxmax()
    return df.loc[idx].sort_values(rank_metric, ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    csv_path = "data_wandb/all_runs_summary.csv"
    if Path(csv_path).exists():
        print(f"Loading from CSV: {csv_path}")
        df = load_from_csv(csv_path)
    else:
        print("Loading from individual summary.json files")
        df = load_all_runs()

    print(f"\nLoaded {len(df)} runs:")
    print(df[["run_name", "model", "encoder", "sim_type", "ndcg@10"]].to_string())
    print(f"\nModels: {sorted(df['model'].unique())}")
    print(f"Encoders: {sorted(df['encoder'].unique())}")
    print(f"Sim types: {sorted(df['sim_type'].unique())}")
    print()
    best = get_best_per_model(df)
    print("Best per model (NDCG@10):")
    print(best[["model", "encoder", "sim_type", "ndcg@10"]].to_string())
