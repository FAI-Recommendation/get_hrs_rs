# Design: Add K=5 alongside K=10 as Primary Evaluation Metric

**Date:** 2026-05-20  
**Status:** Approved

---

## Context

Dataset stats: 553 users, 2194 items, test set = 2105 interactions (~3.8 items/user average).  
At K=10 we recommend 2.6× the average ground truth size → recall inflates, discrimination between models weakens.  
K=5 (1.3× ground truth) is more diagnostic for this small dataset.  
Both K=5 and K=10 are kept: K=10 for comparability with published papers (BM3, FREEDOM, CombiGCN), K=5 for sharper model discrimination.

---

## Scope

4 files changed. No backward-breaking changes — all existing defaults preserved.

---

## Changes per file

### `plot_tier1.py`

1. `summary_table()` — add K=5 columns to display:  
   Change `for k in [10, 20]` → `for k in [5, 10, 20]`

2. `plot_best_vs_best()` — default unchanged (`rank_metric="ndcg@10"`).  
   No code change needed here; callers pass `rank_metric="ndcg@5"` when needed.

### `plot_tier2.py`

1. `plot_ablation()` — add `plot_barplot_per_model(df, model, metric="ndcg", k=5)` immediately before the existing `k=10` barplot call.  
   Both barplots render per model.

2. Print statement in `plot_ablation()` — update to mention both K=5 and K=10:  
   `"Ranking table — {model.upper()} (by NDCG@5 and NDCG@10):"`

### `evaluate.ipynb`

Add a new markdown cell + code cell before the existing Best-vs-Best section:

```
## Tầng 1 — Best-vs-Best @ K=5 (phù hợp hơn cho dataset nhỏ ~3.8 items/user)
best_df5, results5 = plot_best_vs_best(df, rank_metric="ndcg@5")
display(display_summary_table(best_df5))
```

Existing K=10 section remains unchanged below it.

### `EVALUATION_GUIDE.md`

1. Add a note in the "Tầng 1 — Best-vs-Best" section explaining why K=5 is run first.  
2. Update the table of best configs to show two tables (K=5 ranking and K=10 ranking).  
3. Add a Q&A entry: "Q: Tại sao chạy cả K=5 và K=10?" with explanation tied to ~3.8 items/user.

---

## What does NOT change

- `load_data.py` — `get_best_per_model()` default stays `ndcg@10`; callers override as needed.
- `K_VALUES = [1, 5, 10, 20]` — no change.
- All existing function signatures — new behavior is additive only.
