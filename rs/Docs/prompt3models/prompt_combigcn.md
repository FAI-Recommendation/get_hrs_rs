# Prompt: CombiGCN Architecture Diagram (Implementation)

Generate a **high-resolution 4K architecture diagram** (3840×2160 or higher) for our **CombiGCN** recommendation model. Use a clean, academic paper style similar to Fig.1 from the CombiGCN paper. Sharp vector-like rendering, minimal text, professional color palette. This diagram will be used as a **figure only** — formulas and explanations will be written separately below it.

## Architecture Layout (bottom-to-top)

```
                           BPR Loss
                             ↑
                       ŷ = E*ᵤᵀ · E*ᵢ
                             ↑
                    E*ᵤ [553×512]   E*ᵢ [2194×512]
                             ↑
                     mean(layer_0 ... layer_4)
                             ↑
               ┌─────────────┼─────────────┐
               │             │             │
          User CF       Item CF       Item Sim
          Branch        Branch        Branch
               │             │             │
               │             │      W @ item_emb
               │             │        (sim_adj)
               │             │             │
               │         item_inter   item_sim
               │             │             │
               │             └──── ⊕ ─────┘
               │                   │
               │              item_next
               │                   │
               └───── concat ──────┘
                         │
                      ego_emb
                         ↑
                    h⁰ᵢ , h⁰ᵤ
                  ID Embeddings
                                      similarity_adj (W)
                                      [2194 × 2194]
                                           ↑
                                    Precomputed .npz
                                    (cosine_sim → threshold)
                                           ↑
                                      Raw Features
                                    (image / text / both)
```

## Component-by-Component Specification

### Bottom-Left: ID Embeddings
- Two embedding blocks:
  - `h⁰ᵢ` Item [2194 × 512] — grid/matrix icon
  - `h⁰ᵤ` User [553 × 512] — grid/matrix icon
- Concatenated: `ego_emb = concat(h⁰ᵢ, h⁰ᵤ) [2747 × 512]`
- Label: "Item / User ID Embeddings (Xavier init)"

### Bottom-Right: Precomputed Similarity Matrix
- Show data source: `items_features.csv`
- Three paths (color-coded by sim_type):
  - **img_only** (blue): `image vectors → cosine_sim → threshold 0.5`
  - **tfidf** (green): `TF-IDF vectors → cosine_sim → threshold 0.5`
  - **multimodal** (orange): `α·text_sim + (1-α)·img_sim`
- All converge to: `similarity_adj (W) [2194 × 2194]`
- Small label: "Precomputed outside model, cached as .npz"
- **Important:** This is computed at DATA layer, NOT inside the model

### Middle: Propagation Layers — 3 Parallel Branches (MAIN SECTION)

Draw **3 vertical branches** side by side, each with **4 stacked GCN layers**. Show that fusion happens **EVERY layer** (not just at the end).

**Left — "User CF Branch" (light blue background):**
- Each layer: takes user rows from `interaction_adj @ ego_emb`
- 4 stacked GCN layer icons
- Label: "User CF"

**Center — "Item CF Branch" (light yellow background):**
- Each layer: takes item rows from `interaction_adj @ ego_emb`
- 4 stacked GCN layer icons
- Label: "Item CF (interaction)"

**Right — "Item Similarity Branch" (light pink background):**
- Each layer: `W @ item_emb_current`
- 4 stacked GCN layer icons
- Label: "Item Sim"
- Show **sim_type selector** (blue/green/orange) entering this branch — W changes depending on sim_type

**Fusion at EVERY layer (KEY FEATURE):**
- Between Item CF and Item Sim branches: **⊕ symbol** at each layer
- `item_next = item_interaction + item_similar`
- Then: `ego_emb = concat(user_next, item_next)` feeds into next layer
- **This is CombiGCN's defining feature** — fusion every layer, not just at the end
- Draw the ⊕ symbols prominently with arrows from both item branches

**After all 4 layers:**
- `final = mean(layer_0, layer_1, layer_2, layer_3, layer_4)`
- Split into: `E*ᵤ [553 × 512]` and `E*ᵢ [2194 × 512]`

### Top: Prediction & Loss
- Box: **"BPR Loss"**
- `ŷ = E*ᵤᵀ · E*ᵢ` (dot product)
- Label: "No contrastive loss (unlike BM3/FREEDOM)"
- Keep simple — just the box and score equation

## Visual Style

- **Colors:**
  - User CF Branch: light blue background
  - Item CF Branch: light yellow background
  - Item Sim Branch: light pink background
  - Fusion ⊕: bold circle, prominently placed between Item CF and Item Sim
  - Precomputed data: gray/dashed border (outside model)
  - sim_type colors: img_only=blue, tfidf=green, multimodal=orange

- **Nodes:**
  - User-Item graph: bipartite with yellow items, blue users
  - Item-Item similarity: only yellow item nodes

- **Embeddings:** grid/matrix icons (dotted rectangles)

- **Key visual emphasis:**
  - The **⊕ fusion at every layer** should be the most prominent feature
  - The **3 parallel branches** should be clearly separated with background colors
  - Show the **feedback loop** — fused output feeds back as input to next layer

- **Background:** white
- **NO comparison tables, NO detailed formulas** in the diagram

## Key Points for Accurate Generation

1. **Precomputed similarity** — W is built OUTSIDE the model (Data layer), not learned
2. **Fusion EVERY layer** — not just at the end (this is different from BM3/FREEDOM)
3. **3 branches** running in parallel, but Item CF and Item Sim merge at each layer
4. **No contrastive loss** — only BPR + L2 regularization
5. **No modal projectors** — raw features are processed into similarity matrix before training
6. **Only 3 sim_types** — img_only, tfidf, multimodal (no "none" variant in this diagram)

## Title
Below the diagram: **"CombiGCN — Dual-Graph Architecture for Fashion Recommendation"**
