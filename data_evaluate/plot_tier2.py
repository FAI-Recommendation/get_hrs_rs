"""
plot_tier2.py — Ablation analysis: per-model breakdown by encoder × sim_type.

Charts:
    1. Heatmap: sim_type × K, side-by-side clip vs mbnv2
    2. Grouped barplot: NDCG@10 by sim_type, grouped by encoder
    3. Ranking table with highlighted best
    4. Histogram: model configs × K (full, clip, mbnv2)
    5. Heatmap: model configs × K (full, clip, mbnv2)
    6. Overview: histogram + heatmap for multiple metrics
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from load_data import METRICS, K_VALUES, _fig_counter, to_results_dict

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)

CHARTS_DIR = Path(r"E:\DoCode\CD2\source\Source\get_hrs_rs\data_evaluate\charts")


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


def _highlight_max(s):
    """Highlight the maximum value in a Series (for Styler)."""
    is_max = s == s.max()
    return ["font-weight: bold; background-color: #c6efce" if v else "" for v in is_max]


# ─────────────────────────────────────────────
# 1. Heatmap: one model, side-by-side encoders
# ─────────────────────────────────────────────

def plot_heatmap_per_model(df: pd.DataFrame, model: str, metric: str = "ndcg",
                           figsize=(14, 5), save_path: str = None):
    """
    Two heatmaps side by side (clip | mbnv2).
    Rows = sim_type, Cols = K@1/5/10/20.
    """
    model_df = df[df["model"] == model]
    encoders = sorted(model_df["encoder"].unique())

    fig, axes = plt.subplots(1, len(encoders), figsize=figsize, sharey=True)
    if len(encoders) == 1:
        axes = [axes]

    all_vals = []
    for enc in encoders:
        enc_df = model_df[model_df["encoder"] == enc].sort_values("sim_type")
        for k in K_VALUES:
            all_vals.extend(enc_df[f"{metric}@{k}"].values)
    vmin, vmax = min(all_vals), max(all_vals)

    for ax, enc in zip(axes, encoders):
        enc_df = model_df[model_df["encoder"] == enc].sort_values("sim_type")
        matrix = []
        labels = []
        for _, row in enc_df.iterrows():
            vals = [row[f"{metric}@{k}"] for k in K_VALUES]
            matrix.append(vals)
            labels.append(row["sim_type"])

        matrix = np.array(matrix)
        sns.heatmap(
            matrix, annot=True, fmt=".4f",
            xticklabels=[f"K={k}" for k in K_VALUES],
            yticklabels=labels,
            cmap="YlOrRd", vmin=vmin, vmax=vmax,
            ax=ax, cbar=(enc == encoders[-1]),
        )
        ax.set_title(f"{model.upper()} — {enc}", fontsize=13, fontweight="bold")

    fig_title = f"{metric.upper()}@K — {model.upper()} ablation"
    fig.suptitle(fig_title, fontsize=15, y=1.02)
    plt.tight_layout()
    _show_and_save(fig, fig_title, save_path)


# ─────────────────────────────────────────────
# 2. Grouped barplot: sim_type × encoder
# ─────────────────────────────────────────────

def plot_barplot_per_model(df: pd.DataFrame, model: str, metric: str = "ndcg",
                           k: int = 10, figsize=(10, 6), save_path: str = None):
    """
    Bar chart: x = sim_type, grouped by encoder.
    Stars on best bar per sim_type.
    """
    model_df = df[df["model"] == model].copy()
    col = f"{metric}@{k}"
    model_df = model_df.sort_values("sim_type")

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=model_df, x="sim_type", y=col, hue="encoder", ax=ax)

    # Star on best per sim_type
    for sim in model_df["sim_type"].unique():
        subset = model_df[model_df["sim_type"] == sim]
        best_val = subset[col].max()
        best_enc = subset.loc[subset[col].idxmax(), "encoder"]

        sim_types_sorted = sorted(model_df["sim_type"].unique())
        encoders_sorted = sorted(model_df["encoder"].unique())
        x_idx = sim_types_sorted.index(sim)
        n_enc = len(encoders_sorted)
        bar_width = 0.8 / n_enc
        enc_idx = encoders_sorted.index(best_enc)
        x_pos = x_idx + (enc_idx - (n_enc - 1) / 2) * bar_width
        ax.plot(x_pos, best_val, "r*", markersize=15, zorder=5)

    fig_title = f"{model.upper()} — {metric.upper()}@{k} by sim_type x encoder"
    ax.set_title(fig_title, fontsize=13, fontweight="bold")
    ax.set_xlabel("sim_type")
    ax.set_ylabel(f"{metric.upper()}@{k}")
    ax.legend(title="Encoder")
    plt.tight_layout()
    _show_and_save(fig, fig_title, save_path)


# ─────────────────────────────────────────────
# 3. Ranking table per model
# ─────────────────────────────────────────────

def ranking_table_per_model(df: pd.DataFrame, model: str,
                            rank_metric: str = "ndcg@10") -> pd.DataFrame:
    """
    Return a styled DataFrame with all metrics for a given model,
    sorted by rank_metric, max values highlighted.
    """
    model_df = df[df["model"] == model].copy()
    display_cols = ["encoder", "sim_type"]
    for m in METRICS:
        for k in K_VALUES:
            display_cols.append(f"{m}@{k}")

    table = model_df[display_cols].sort_values(rank_metric, ascending=False)
    table = table.reset_index(drop=True)
    table.index = table.index + 1  # 1-based ranking
    table.index.name = "Rank"
    return table


def display_ranking_table(df: pd.DataFrame, model: str,
                          rank_metric: str = "ndcg@10"):
    """Print the ranking table with highlighted max values."""
    table = ranking_table_per_model(df, model, rank_metric)
    metric_cols = [c for c in table.columns if "@" in c]
    styled = table.style.apply(_highlight_max, subset=metric_cols)
    return styled


# ─────────────────────────────────────────────
# All-in-one per model
# ─────────────────────────────────────────────

def plot_ablation(df: pd.DataFrame, models: list = None,
                  metrics_to_heatmap: list = None, save_dir: str = None):
    """
    Run full Tier 2 ablation for specified models.
    - Heatmap for each metric
    - Barplot NDCG@10
    - Print ranking table
    """
    if models is None:
        models = sorted(df["model"].unique())
    if metrics_to_heatmap is None:
        metrics_to_heatmap = ["ndcg", "recall", "precision", "hit_ratio", "map", "mrr"]

    for model in models:
        print(f"\n{'='*60}")
        print(f"  ABLATION: {model.upper()}")
        print(f"{'='*60}\n")

        for metric in metrics_to_heatmap:
            sp = f"{save_dir}/{model}_heatmap_{metric}.png" if save_dir else None
            plot_heatmap_per_model(df, model, metric=metric, save_path=sp)

        sp = f"{save_dir}/{model}_barplot_ndcg5.png" if save_dir else None
        plot_barplot_per_model(df, model, metric="ndcg", k=5, save_path=sp)

        sp = f"{save_dir}/{model}_barplot_ndcg10.png" if save_dir else None
        plot_barplot_per_model(df, model, metric="ndcg", k=10, save_path=sp)

        table = ranking_table_per_model(df, model)
        print(f"\nRanking table — {model.upper()} (by NDCG@5 và NDCG@10):")
        print(table[["encoder", "sim_type"] +
                     [f"{m}@10" for m in METRICS]].to_string())
        print()


# ─────────────────────────────────────────────
# 4. Histogram: horizontal grouped bar (model × K)
# ─────────────────────────────────────────────

def plot_histogram_models(df: pd.DataFrame, encoder: str = None,
                          metric: str = "ndcg",
                          figsize=None, save_path: str = None,
                          title_suffix: str = ""):
    """
    Horizontal grouped bar chart.
    Y-axis = model config labels, X-axis = metric value, hue = K value.

    encoder: None → all models, 'clip' → clip only, 'mbnv2' → mbnv2 only.
    """
    plot_df = df.copy()
    if encoder is not None:
        plot_df = plot_df[plot_df["encoder"] == encoder]

    plot_df = plot_df.copy()
    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['model']}_{r['encoder']}({r['sim_type']})", axis=1
    )

    # Sort order: best ndcg@10 at left
    sort_order = (
        plot_df.sort_values(f"{metric}@10", ascending=False)["label"].tolist()
    )

    # Melt to long format for seaborn
    k_cols = [f"{metric}@{k}" for k in K_VALUES]
    melted = plot_df[["label"] + k_cols].melt(
        id_vars="label", var_name="K_str", value_name="value"
    )
    melted["K"] = melted["K_str"].str.extract(r"@(\d+)").astype(int).astype(str)
    melted["K"] = "K=" + melted["K"]

    n_models = len(sort_order)
    if figsize is None:
        figsize = (max(14, n_models * 0.9), 7)

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("Set2", n_colors=len(K_VALUES))

    sns.barplot(
        data=melted, x="label", y="value", hue="K",
        order=sort_order, hue_order=[f"K={k}" for k in K_VALUES],
        palette=palette, ax=ax,
    )

    title = f"{metric.upper()}@K — All model configs"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{metric.upper()} value")
    ax.set_xlabel("Model config")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.legend(title="K", loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    _show_and_save(fig, title, save_path)


def plot_histogram_overview(df: pd.DataFrame, metric: str = "ndcg",
                            save_dir: str = None):
    """
    Plot 3 histograms:
      1. Full — all 24 models
      2. CLIP only — 12 models
      3. MBNv2 only — 12 models
    """
    configs = [
        (None,     "All encoders", "hist_full"),
        ("clip",   "CLIP",         "hist_clip"),
        ("mbnv2",  "MBNv2",        "hist_mbnv2"),
    ]
    for enc, suffix, fname in configs:
        sp = f"{save_dir}/{fname}_{metric}.png" if save_dir else None
        plot_histogram_models(df, encoder=enc, metric=metric,
                              title_suffix=suffix, save_path=sp)


# ─────────────────────────────────────────────
# 5. Heatmap: model configs × K (all/clip/mbnv2)
# ─────────────────────────────────────────────

def plot_heatmap_models(df: pd.DataFrame, encoder: str = None,
                        metric: str = "ndcg",
                        figsize=None, save_path: str = None,
                        title_suffix: str = ""):
    """
    Heatmap with rows = model configs, cols = K values.

    encoder: None → all models, 'clip' → clip only, 'mbnv2' → mbnv2 only.
    """
    plot_df = df.copy()
    if encoder is not None:
        plot_df = plot_df[plot_df["encoder"] == encoder]

    if encoder is None:
        plot_df["label"] = plot_df.apply(
            lambda r: f"{r['model']}_{r['encoder']}({r['sim_type']})", axis=1
        )
    else:
        plot_df["label"] = plot_df.apply(
            lambda r: f"{r['model']}({r['sim_type']})", axis=1
        )

    sort_order = (
        plot_df.sort_values(f"{metric}@10", ascending=False)["label"].tolist()
    )

    k_cols = [f"{metric}@{k}" for k in K_VALUES]
    matrix = plot_df.set_index("label").loc[sort_order, k_cols].values

    n_models = len(sort_order)
    if figsize is None:
        figsize = (8, max(6, n_models * 0.45))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        xticklabels=[f"K={k}" for k in K_VALUES],
        yticklabels=sort_order,
        cmap="YlOrRd",
        ax=ax,
        cbar=True,
        annot_kws={"size": 8},
    )

    title = f"{metric.upper()}@K — Model configs heatmap"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Model config")
    plt.tight_layout()
    _show_and_save(fig, title, save_path)


def plot_heatmap_overview(df: pd.DataFrame, metric: str = "ndcg",
                          save_dir: str = None):
    """
    Plot 3 heatmaps:
      1. Full — all 24 models
      2. CLIP only — 12 models
      3. MBNv2 only — 12 models
    """
    configs = [
        (None,     "All encoders", "heatmap_full"),
        ("clip",   "CLIP",         "heatmap_clip"),
        ("mbnv2",  "MBNv2",        "heatmap_mbnv2"),
    ]
    for enc, suffix, fname in configs:
        sp = f"{save_dir}/{fname}_{metric}.png" if save_dir else None
        plot_heatmap_models(df, encoder=enc, metric=metric,
                            title_suffix=suffix, save_path=sp)


# ─────────────────────────────────────────────
# 6. Lineplot: model configs × K (all/clip/mbnv2)
# ─────────────────────────────────────────────

def plot_lineplot_models(df: pd.DataFrame, encoder: str = None,
                         metric: str = "ndcg",
                         figsize=(14, 7), save_path: str = None,
                         title_suffix: str = ""):
    """
    Line plot: x = K, y = metric, one line per model config.
    Red star on the best line at each K.

    encoder: None → all models, 'clip' → clip only, 'mbnv2' → mbnv2 only.
    """
    plot_df = df.copy()
    if encoder is not None:
        plot_df = plot_df[plot_df["encoder"] == encoder]

    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['model']}_{r['encoder']}({r['sim_type']})", axis=1
    )

    colors = plt.cm.tab20.colors
    line_styles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=figsize)

    # Find best label per K → set of all winning labels (may be 1 or more models)
    k_cols = [f"{metric}@{k}" for k in K_VALUES]
    max_per_k = plot_df[k_cols].max()
    winning_labels = set()
    for k in K_VALUES:
        col = f"{metric}@{k}"
        winning_labels.add(plot_df.loc[plot_df[col].idxmax(), "label"])

    for i, (_, row) in enumerate(plot_df.iterrows()):
        label = row["label"]
        vals = [row[f"{metric}@{k}"] for k in K_VALUES]
        color = colors[i % len(colors)]
        ls = line_styles[(i // len(colors)) % len(line_styles)]
        is_winner = label in winning_labels

        ax.plot(K_VALUES, vals, linestyle=ls, marker="o",
                label=label, color=color,
                linewidth=2.8 if is_winner else 1.0,
                markersize=8 if is_winner else 4,
                alpha=1.0 if is_winner else 0.35,
                zorder=3 if is_winner else 2)

        # Star + text annotation at each K where this line wins
        for k, v in zip(K_VALUES, vals):
            if v == max_per_k[f"{metric}@{k}"]:
                ax.plot(k, v, "r*", markersize=14, zorder=5)
                short = label.split("(")[0]
                ax.annotate(
                    short,
                    xy=(k, v),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=7.5, color="darkred", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
                    zorder=6,
                )

    title = f"{metric.upper()}@K — All model configs (line)"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(K_VALUES)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5, ncol=1)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    _show_and_save(fig, title, save_path)


def plot_lineplot_overview(df: pd.DataFrame, metric: str = "ndcg",
                           save_dir: str = None):
    """
    Plot 3 line plots (full / CLIP / MBNv2) for one metric.
    """
    configs = [
        (None,    "All encoders", "lineplot_full"),
        ("clip",  "CLIP",         "lineplot_clip"),
        ("mbnv2", "MBNv2",        "lineplot_mbnv2"),
    ]
    for enc, suffix, fname in configs:
        sp = f"{save_dir}/{fname}_{metric}.png" if save_dir else None
        plot_lineplot_models(df, encoder=enc, metric=metric,
                             title_suffix=suffix, save_path=sp)


def plot_overview_all_metrics(df: pd.DataFrame, metrics: list = None,
                              save_dir: str = None):
    """
    Plot histogram + heatmap + lineplot overview for multiple metrics.

    Default metrics: ndcg, recall, precision, map, mrr.
    Each metric includes 24 models (full) and 12 models for CLIP/MBNv2.
    """
    if metrics is None:
        metrics = ["ndcg", "recall", "precision", "map", "mrr"]

    for metric in metrics:
        plot_histogram_overview(df, metric=metric, save_dir=save_dir)
        plot_heatmap_overview(df, metric=metric, save_dir=save_dir)
        plot_lineplot_overview(df, metric=metric, save_dir=save_dir)


# ─────────────────────────────────────────────
# 7. Custom 1×3 lineplot: recall + precision + ndcg, split by encoder
# ─────────────────────────────────────────────

def plot_lineplot_3metrics(df: pd.DataFrame, encoder: str,
                           metrics: list = None,
                           figsize=(18, 6), save_path: str = None):
    """
    1×3 figure — each subplot = one metric (recall, precision, ndcg).
    Lines = 12 model configs for the given encoder (clip or mbnv2).
    Red star ★ on best value per K. Legend on the right of the figure.

    encoder: 'clip' or 'mbnv2'
    """
    if metrics is None:
        metrics = ["recall", "precision", "ndcg"]

    plot_df = df[df["encoder"] == encoder].copy()
    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['model']}({r['sim_type']})", axis=1
    )

    colors = plt.cm.tab20.colors
    line_styles = ["-", "--", "-.", ":"]
    labels = plot_df["label"].tolist()

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=False)

    for ax, metric in zip(axes, metrics):
        k_cols = [f"{metric}@{k}" for k in K_VALUES]
        max_per_k = plot_df[k_cols].max()

        # Labels that win at any K for this metric
        winning_labels = set()
        for k in K_VALUES:
            winning_labels.add(
                plot_df.loc[plot_df[f"{metric}@{k}"].idxmax(), "label"]
            )

        for i, (_, row) in enumerate(plot_df.iterrows()):
            label = row["label"]
            vals = [row[f"{metric}@{k}"] for k in K_VALUES]
            color = colors[i % len(colors)]
            ls = line_styles[(i // len(colors)) % len(line_styles)]
            is_winner = label in winning_labels

            ax.plot(K_VALUES, vals, linestyle=ls, marker="o",
                    label=label, color=color,
                    linewidth=2.8 if is_winner else 1.0,
                    markersize=8 if is_winner else 4,
                    alpha=1.0 if is_winner else 0.35,
                    zorder=3 if is_winner else 2)

            for k, v in zip(K_VALUES, vals):
                if v == max_per_k[f"{metric}@{k}"]:
                    ax.plot(k, v, "r*", markersize=15, zorder=5)
                    ax.annotate(
                        label,
                        xy=(k, v),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=7.5, color="darkred", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
                        zorder=6,
                    )

        ax.set_title(f"{metric.upper()}@K", fontsize=13, fontweight="bold")
        ax.set_xlabel("K")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(K_VALUES)
        ax.grid(True, linestyle="--", alpha=0.7)

    # Shared legend outside the figure on the right
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels,
               loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=9, framealpha=0.9)
    fig_title = f"Recall / Precision / NDCG@K — {encoder.upper()}"
    fig.suptitle(fig_title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _show_and_save(fig, fig_title, save_path)


def plot_lineplot_3metrics_both(df: pd.DataFrame, metrics: list = None,
                                save_dir: str = None):
    """
    2 figures: one for CLIP (12 models), one for MBNv2 (12 models).
    Each figure has 3 subplots: recall, precision, ndcg.
    """
    for encoder in ["clip", "mbnv2"]:
        sp = f"{save_dir}/lineplot_3metrics_{encoder}.png" if save_dir else None
        plot_lineplot_3metrics(df, encoder=encoder, metrics=metrics, save_path=sp)


# ─────────────────────────────────────────────
# 8. Radar overview: full / clip / mbnv2
# ─────────────────────────────────────────────

_RADAR_METRICS = ["recall", "precision", "ndcg", "hit_ratio", "map", "mrr"]


def _plot_single_radar(results_dict: dict, title: str,
                       metrics: list = None,
                       figsize=(10, 10), save_path: str = None):
    """
    Radar chart for an arbitrary results_dict (any number of model configs).
    Each line = one config, values = mean across K for each metric.
    """
    if metrics is None:
        metrics = _RADAR_METRICS

    avg = {}
    for label, data in results_dict.items():
        avg[label] = {m: float(np.mean(data[m])) for m in metrics if m in data}

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles_plot = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))
    colors = plt.cm.tab20.colors

    for i, (label, m_vals) in enumerate(avg.items()):
        values = [m_vals.get(m, 0) for m in metrics] + [m_vals.get(metrics[0], 0)]
        ax.plot(angles_plot, values, "o-", linewidth=1.5,
                label=label, color=colors[i % len(colors)], alpha=0.8)
        ax.fill(angles_plot, values, alpha=0.05, color=colors[i % len(colors)])

    ax.set_xticks(angles)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    n = len(results_dict)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=8 if n > 12 else 9,
        ncol=2 if n > 12 else 1,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    _show_and_save(fig, title, save_path)


def plot_radar_overview(df: pd.DataFrame, metrics: list = None,
                        save_dir: str = None):
    """
    3 radar charts showing all model configs grouped by encoder scope:
      1. Full  — all 24 configs
      2. CLIP  — 12 configs
      3. MBNv2 — 12 configs

    Call this before plot_best_vs_best() so the best-vs-best radar stays last.
    """
    configs = [
        (None,    "Radar Overview — All 24 Configs",    "radar_overview_full"),
        ("clip",  "Radar Overview — CLIP (12 Configs)", "radar_overview_clip"),
        ("mbnv2", "Radar Overview — MBNv2 (12 Configs)","radar_overview_mbnv2"),
    ]
    for enc, title, fname in configs:
        sub = df if enc is None else df[df["encoder"] == enc]
        results = to_results_dict(sub)
        sp = f"{save_dir}/{fname}.png" if save_dir else None
        _plot_single_radar(results, title, metrics=metrics, save_path=sp)
