"""
plot_tier1.py — Best-vs-best model comparison.

Charts:
    1. 5-metric lineplot grid with red stars on best
    2. Radar chart: average metric per model
    3. Summary table with highlighted winner
    4. Preview lineplot: recall/precision/ndcg/hit_ratio (best config per model)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import METRICS, K_VALUES, to_results_dict, get_best_per_model, _fig_counter

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)

CHARTS_DIR = Path(r"E:\DoCode\CD2\source\Source\get_hrs_rs\data_evaluate\charts_v2")


def _show_and_save(fig, title: str, save_path: str = None):
    """Print numbered figure title, auto-save to CHARTS_DIR, then show."""
    _fig_counter[0] += 1
    n = _fig_counter[0]
    full_title = f"Figure {n:02d}. {title}"
    print(f"\n── {full_title} ──")
    slug = re.sub(r"[^\w\-]+", "_", full_title).strip("_")
    out = save_path or str(CHARTS_DIR / f"{slug}.png")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()

_DISPLAY_METRICS = ["recall", "precision", "ndcg", "hit_ratio", "map", "mrr"]


def _highlight_max(s):
    is_max = s == s.max()
    return ["font-weight: bold; background-color: #c6efce" if v else "" for v in is_max]


# ─────────────────────────────────────────────
# 1. 5-metric lineplot grid (like reference image)
# ─────────────────────────────────────────────

def plot_lineplot_grid(results_dict: dict, metrics: list = None,
                       k_values: list = None, figsize=(20, 12),
                       save_path: str = None):
    """
    2×3 subplot grid. Each subplot = one metric.
    Lines = models, x = K, y = metric value.
    Red stars on best per K.
    """
    if metrics is None:
        metrics = _DISPLAY_METRICS
    if k_values is None:
        k_values = K_VALUES

    n_metrics = len(metrics)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs_flat = axs.flatten() if hasattr(axs, "flatten") else [axs]

    colors = plt.cm.tab10.colors
    line_styles = ["-", "--", "-.", ":"]

    for idx, metric in enumerate(metrics):
        ax = axs_flat[idx]

        # Find max per K
        max_values = []
        for k_idx in range(len(k_values)):
            valid = [v[metric][k_idx] for v in results_dict.values()
                     if metric in v]
            max_values.append(max(valid) if valid else 0)

        # All labels that win at any K
        winning_labels = set()
        for k_idx, max_val in enumerate(max_values):
            for lbl, data in results_dict.items():
                if metric in data and data[metric][k_idx] == max_val:
                    winning_labels.add(lbl)

        for i, (label, data) in enumerate(results_dict.items()):
            if metric not in data:
                continue
            color = colors[i % len(colors)]
            ls = line_styles[(i // len(colors)) % len(line_styles)]
            is_winner = label in winning_labels

            ax.plot(k_values, data[metric], linestyle=ls, marker="o",
                    label=label, color=color,
                    linewidth=2.8 if is_winner else 1.0,
                    markersize=8 if is_winner else 4,
                    alpha=1.0 if is_winner else 0.35,
                    zorder=3 if is_winner else 2)

            for k_idx, val in enumerate(data[metric]):
                if val == max_values[k_idx]:
                    ax.plot(k_values[k_idx], val, "r*", markersize=15, zorder=5)
                    ax.annotate(
                        label,
                        xy=(k_values[k_idx], val),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=7.5, color="darkred", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
                        zorder=6,
                    )

        ax.set_title(f"{metric.upper()}@K", fontsize=13, fontweight="bold")
        ax.set_xlabel("K")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(k_values)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.7)

    for i in range(n_metrics, len(axs_flat)):
        fig.delaxes(axs_flat[i])

    fig.suptitle("Best-vs-Best Model Comparison", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    _show_and_save(fig, "Best-vs-Best Model Comparison", save_path)


# ─────────────────────────────────────────────
# 1b. Single metric lineplot (enlarged)
# ─────────────────────────────────────────────

def plot_lineplot_single(results_dict: dict, metric: str = "ndcg",
                         k_values: list = None, figsize=(12, 8),
                         save_path: str = None):
    """Single metric lineplot, larger and cleaner."""
    if k_values is None:
        k_values = K_VALUES

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
    line_styles = ["-", "--", "-.", ":"]

    max_values = []
    for k_idx in range(len(k_values)):
        valid = [v[metric][k_idx] for v in results_dict.values() if metric in v]
        max_values.append(max(valid) if valid else 0)

    # All labels that win at any K
    winning_labels = set()
    for k_idx, max_val in enumerate(max_values):
        for lbl, data in results_dict.items():
            if metric in data and data[metric][k_idx] == max_val:
                winning_labels.add(lbl)

    for i, (label, data) in enumerate(results_dict.items()):
        if metric not in data:
            continue
        color = colors[i % len(colors)]
        ls = line_styles[(i // len(colors)) % len(line_styles)]
        is_winner = label in winning_labels

        ax.plot(k_values, data[metric], linestyle=ls, marker="o",
                label=label, color=color,
                linewidth=2.8 if is_winner else 1.0,
                markersize=9 if is_winner else 5,
                alpha=1.0 if is_winner else 0.35,
                zorder=3 if is_winner else 2)

        for k_idx, val in enumerate(data[metric]):
            if val == max_values[k_idx]:
                ax.plot(k_values[k_idx], val, "r*", markersize=18, zorder=5)
                ax.annotate(
                    label,
                    xy=(k_values[k_idx], val),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color="darkred", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
                    zorder=6,
                )

    fig_title = f"Best-vs-Best — {metric.upper()}@K"
    ax.set_title(fig_title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("K", fontsize=13)
    ax.set_ylabel(metric.upper(), fontsize=13)
    ax.set_xticks(k_values)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    _show_and_save(fig, fig_title, save_path)


# ─────────────────────────────────────────────
# 2. Radar chart
# ─────────────────────────────────────────────

def plot_radar_chart(results_dict: dict, metrics: list = None,
                     figsize=(10, 10), save_path: str = None):
    """Radar chart: average across K for each metric, per model."""
    if metrics is None:
        metrics = _DISPLAY_METRICS

    avg = {}
    for label, data in results_dict.items():
        avg[label] = {}
        for m in metrics:
            if m in data:
                avg[label][m] = np.mean(data[m])

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))
    colors = plt.cm.tab10.colors

    for i, (label, m_vals) in enumerate(avg.items()):
        values = [m_vals.get(m, 0) for m in metrics]
        values += values[:1]
        angles_plot = np.concatenate((angles, [angles[0]]))

        ax.plot(angles_plot, values, "o-", linewidth=2, label=label,
                color=colors[i % len(colors)])
        ax.fill(angles_plot, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Overall Model Performance", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    _show_and_save(fig, "Overall Model Performance — Radar", save_path)


# ─────────────────────────────────────────────
# 3. Summary table
# ─────────────────────────────────────────────

def summary_table(df_best: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table from best-per-model DataFrame.
    Columns: model, encoder, sim_type, key metrics.
    """
    display_cols = ["model", "encoder", "sim_type"]
    for m in _DISPLAY_METRICS:
        for k in [5, 10, 20]:
            display_cols.append(f"{m}@{k}")

    table = df_best[display_cols].copy()
    table = table.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    table.index = table.index + 1
    table.index.name = "Rank"
    return table


def display_summary_table(df_best: pd.DataFrame):
    """Return styled summary table with highlighted max."""
    table = summary_table(df_best)
    metric_cols = [c for c in table.columns if "@" in c]
    return table.style.apply(_highlight_max, subset=metric_cols)


# ─────────────────────────────────────────────
# 4. Histogram: best config per model × 4 metrics
# ─────────────────────────────────────────────

_HIST_METRICS = ["recall", "precision", "ndcg", "hit_ratio", "map", "mrr"]


def plot_histogram_tier1(df: pd.DataFrame,
                         metrics: list = None,
                         rank_metric: str = "ndcg@10",
                         figsize=(20, 12),
                         save_path: str = None):
    """
    2×3 grid — one subplot per metric (recall, precision, ndcg, hit_ratio, map, mrr).
    Each subplot: x = best-config label per model, grouped bars = K=1/5/10/20.
    """
    if metrics is None:
        metrics = _HIST_METRICS

    best = get_best_per_model(df, rank_metric=rank_metric)
    best = best.copy()
    best["label"] = best.apply(
        lambda r: f"{r['model']}_{r['encoder']}({r['sim_type']})", axis=1
    )

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten()
    palette = sns.color_palette("Set2", n_colors=len(K_VALUES))

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        sort_order = (
            best.sort_values(f"{metric}@10", ascending=False)["label"].tolist()
        )

        k_cols = [f"{metric}@{k}" for k in K_VALUES]
        melted = best[["label"] + k_cols].melt(
            id_vars="label", var_name="K_str", value_name="value"
        )
        melted["K"] = "K=" + melted["K_str"].str.extract(r"@(\d+)")[0]

        sns.barplot(
            data=melted, x="label", y="value", hue="K",
            order=sort_order, hue_order=[f"K={k}" for k in K_VALUES],
            palette=palette, ax=ax,
        )
        ax.set_title(f"{metric.upper()}@K", fontsize=13, fontweight="bold")
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        ax.legend(title="K", fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for i in range(len(metrics), len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.suptitle("Tier 1 — Best Config per Model: Metrics Overview",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _show_and_save(fig, "Tier 1 — Best Config per Model Metrics Overview", save_path)
    return best


def plot_tier1_metrics_preview(df: pd.DataFrame,
                               rank_metric: str = "ndcg@10",
                               metrics: list = None,
                               figsize=(18, 12),
                               save_path: str = None):
    """
    Backward-compatible wrapper for Tier 1 line preview metrics.
    Default metrics: recall, precision, ndcg, hit_ratio.
    """
    return plot_tier1_line_preview(
        df,
        rank_metric=rank_metric,
        metrics=metrics,
        figsize=figsize,
        save_path=save_path,
    )


def plot_tier1_line_preview(df: pd.DataFrame,
                            rank_metric: str = "ndcg@10",
                            metrics: list = None,
                            figsize=(18, 12),
                            save_path: str = None):
    """
    Preview lineplots for recall/precision/ndcg/hit_ratio before best-vs-best.
    Uses best config per model (ranked by rank_metric).
    """
    if metrics is None:
        metrics = _HIST_METRICS

    best = get_best_per_model(df, rank_metric=rank_metric)
    results = to_results_dict(best)
    plot_lineplot_grid(results, metrics=metrics, figsize=figsize, save_path=save_path)
    return best, results


# ─────────────────────────────────────────────
# All-in-one
# ─────────────────────────────────────────────

def plot_best_vs_best(df: pd.DataFrame, rank_metric: str = "ndcg@10",
                      save_dir: str = None):
    """
    Full Tier 1 analysis: select best config per model, then compare.
    """
    best = get_best_per_model(df, rank_metric=rank_metric)

    print(f"\n{'='*60}")
    print(f"  TIER 1: BEST-vs-BEST (ranked by {rank_metric})")
    print(f"{'='*60}\n")

    print("Selected configs:")
    for _, row in best.iterrows():
        print(f"  {row['model'].upper():12s} → {row['encoder']}({row['sim_type']})  "
              f"{rank_metric}={row[rank_metric]:.6f}")
    print()

    results = to_results_dict(best)

    # Lineplot grid
    sp = f"{save_dir}/tier1_lineplot_grid.png" if save_dir else None
    plot_lineplot_grid(results, save_path=sp)

    # Single lineplot for NDCG
    sp = f"{save_dir}/tier1_ndcg_lineplot.png" if save_dir else None
    plot_lineplot_single(results, metric="ndcg", save_path=sp)

    # Radar
    sp = f"{save_dir}/tier1_radar.png" if save_dir else None
    plot_radar_chart(results, save_path=sp)

    # Table
    table = summary_table(best)
    print("\nSummary table (best-vs-best):")
    print(table.to_string())
    print()

    return best, results


# ─────────────────────────────────────────────
# Best overall model per metric (bar chart)
# ─────────────────────────────────────────────

_BEST_OVERALL_METRICS = ["recall", "precision", "ndcg", "hit_ratio", "mrr", "map"]


def plot_best_overall_per_metric(df: pd.DataFrame,
                                 metrics: list = None,
                                 figsize=(14, 7),
                                 save_path: str = None):
    """
    Bar chart: for each metric, show the model config with the highest
    mean score averaged across all K values.
    X = metrics, Y = mean score. Each bar annotated with value + config name.
    """
    if metrics is None:
        metrics = _BEST_OVERALL_METRICS

    best_labels = []
    best_scores = []

    for metric in metrics:
        k_cols = [f"{metric}@{k}" for k in K_VALUES]
        df_tmp = df.copy()
        df_tmp["_mean"] = df_tmp[k_cols].mean(axis=1)
        best_row = df_tmp.loc[df_tmp["_mean"].idxmax()]
        label = f"{best_row['model']}_{best_row['encoder']}({best_row['sim_type']})"
        best_labels.append(label)
        best_scores.append(float(best_row["_mean"]))

    colors = plt.cm.tab10.colors
    bar_colors = [colors[i % len(colors)] for i in range(len(metrics))]

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = range(len(metrics))
    bars = ax.bar(
        x_pos, best_scores,
        color=bar_colors, edgecolor="black", linewidth=0.6,
        width=0.55,
    )

    y_pad = max(best_scores) * 0.015
    for bar, label, score in zip(bars, best_labels, best_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_pad,
            f"{label}\n{score:.4f}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="black",
        )

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_title("Best Overall Models for Each Metric", fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    # extra headroom for annotations
    ax.set_ylim(0, max(best_scores) * 1.35)
    plt.tight_layout()
    _show_and_save(fig, "Best Overall Models for Each Metric", save_path)
