# Prompt: BM3 Architecture Diagram (Adapted Implementation)

Generate a **high-resolution 4K architecture diagram** (3840×2160 or higher) for our **adapted BM3 (Bootstrap Multi-modal Model)** recommendation model. Use a clean, academic paper style. Sharp vector-like rendering, minimal text, professional color palette. This diagram will be used as a **figure only** — formulas and explanations will be written separately below it.

**IMPORTANT: This is our ADAPTED implementation, NOT the exact paper. Follow this spec exactly.**

## Architecture Layout (bottom-to-top)

```
                              BPR Loss + Bootstrap CL Loss
                                ↑              ↑
                          ŷ = hᵤᵀ · hᵢ    Bootstrap CL
                                ↑              ↑
                          hᵢ = item_cf + modal_emb
                               ↑           ↑
                    ┌──────────┘           └──────────┐
                    │                                  │
               item_cf                            modal_emb
               user_cf                          [2194 × 512]
                    ↑                                  ↑
             [CF Branch]                     [Modal Branch]
           GCN Propagation               Linear Projectors
                    ↑                                  ↑
           User-Item Graph                      Raw Features
           + Dropout                           (image / text)
                    ↑
              h⁰ᵢ , h⁰ᵤ
            ID Embeddings
                    ↑
          h⁰ᵢ_target (EMA copy, frozen)
```

## Component-by-Component Specification

### Bottom-Left: ID Embeddings
- Two embedding blocks:
  - `h⁰ᵢ` Item [2194 × 512] — grid/matrix icon
  - `h⁰ᵤ` User [553 × 512] — grid/matrix icon
- Below item embedding: **`h⁰ᵢ_target`** — a copy with a **🔄 circular arrow** icon
  - Label: "EMA Target (frozen, momentum update only)"
  - Visually show it as a shadowed/ghost copy of h⁰ᵢ
- Label: "Item / User ID Embeddings"

### Bottom-Right: Raw Features + Linear Projection
- Two input sources:
  - `image_feats [2194 × 512]` — image icon
  - `text_feats [2194 × 768]` — text icon
- Arrow into a box labeled **"Linear Projectors"** (NOT MLP — single layer, no activation)
- Show **sim_type selector** with 4 variants:
  - img_only (blue): proj_img only
  - tfidf (green): proj_txt only
  - multimodal (orange): (proj_img + proj_txt) / 2
  - mm_attention (red): Linear(concat)
- Output: ONE `modal_emb [2194 × 512]`

### Left Branch: CF Branch (light blue background)
- Label the branch as **"CF Branch"** (do NOT write "LightGCN" — this is BM3's own GCN propagation)
- Input: `ego = concat(h⁰ᵢ, h⁰ᵤ) [2747 × 512]`
- User-Item bipartite graph (2 layers of nodes, users + items)
- **"Dropout"** label (random edge dropout during training)
- 4 stacked GCN layers: `ego = interaction_adj @ ego`
- Draw as stacked circle layers
- Output split: `user_cf [553 × 512]` + `item_cf [2194 × 512]`

### Right Branch: Modal Branch (light orange background)
- **NO graph propagation** — this is the KEY difference from FREEDOM
- modal_emb goes DIRECTLY to fusion (no GCN layers)
- Just the Linear Projectors → modal_emb output
- Label: "Direct projection (no graph propagation)"

### Middle: Fusion + Bootstrap Contrastive Learning

This section combines fusion AND the bootstrap CL — draw them together to avoid duplicating `item_cf`.

**Fusion:**
- Large **(+)** circle
- `hᵢ = item_cf + modal_emb`
- `item_cf` arrow comes from CF Branch (left), `modal_emb` arrow comes from Modal Branch (right)
- **IMPORTANT: `item_cf` should appear ONCE only** — it feeds into both fusion AND bootstrap CL from the same single node

**Bootstrap Contrastive Learning (light yellow background — MOST PROMINENT):**

This is BM3's **defining feature**. Draw it as a **V-shape** (NOT a triangle — there are only 2 edges, NOT 3).

```
     item_cf ←── L_boot ──→ modal_emb ←── L_boot ──→ item_target
                                                       (EMA encoder)
```

The shape is a **V** with `modal_emb` at the center:
- **Left node**: `item_cf` (the SAME node that feeds into fusion — do NOT draw a second item_cf)
- **Center node**: `modal_emb` (from Linear Projectors)
- **Right node**: `item_target` (from EMA target encoder, with 🔄 icon)

Connect them with **exactly 2 edges** (NOT 3):
- Edge 1: item_cf ↔ modal_emb — label **"L_boot"**
- Edge 2: modal_emb ↔ item_target — label **"L_boot"**
- **NO edge between item_cf and item_target** — they are NOT directly connected

**Predictor Head** — draw as a small separate box NEAR the item_cf node:
- Label: **"Predictor"**
- Small subtitle: "asymmetric — prevents collapse"
- Do NOT write the internal formula — it will be explained below the figure

**EMA Update** — circular arrow icon near item_target:
- Label: **"EMA momentum = 0.995"**

**IMPORTANT: Do NOT put any formulas inside the Bootstrap CL section.** Only node names, edge labels ("L_boot"), and short labels. All detailed formulas will be written separately below the figure.

**CRITICAL: Do NOT draw a triangle. There are exactly 2 L_boot edges, forming a V-shape through modal_emb. There is NO direct connection between item_cf and item_target.**

### Very Top: Loss
- Box: **"BPR Loss + Bootstrap CL Loss"**
- Arrows from `hᵤ`, `hᵢ` → BPR
- Arrows from Bootstrap CL V-shape → Bootstrap CL Loss

## Visual Style

- **Colors:**
  - CF Branch: light blue background
  - Modal Branch: light orange background
  - Bootstrap CL section: light yellow background — **give it the MOST space** (it is the defining feature of BM3)
  - Linear Projectors: solid blue box with white text
  - Predictor: solid green box with white text
  - EMA Target: dashed border (frozen)
  - Fusion (+): white circle with black border

- **Nodes in Bootstrap CL triangle:**
  - Make them **LARGE** (at least 3x bigger than GCN layer circles)
  - Each node has a clear label inside or next to it
  - Edges between nodes are thick with "L_boot" label in readable font size

- **General nodes:**
  - Item nodes: yellow/gold circles
  - User nodes: blue circles

- **Embeddings:** grid/matrix icons (dotted rectangles)

- **Key icons:**
  - 🔄 circular arrow on EMA target
  - Dashed lines for frozen/detached components

- **Background:** white

- **CRITICAL: NO formulas anywhere in the diagram. NO small text. All text must be readable at a glance.** Formulas and detailed explanations will be written separately below the figure.

## Key Points for Accurate Generation

1. **Do NOT label the CF Branch as "LightGCN"** — call it "CF Branch" or "GCN Propagation"
2. **Linear Projectors** (NOT MLP) — single layer `nn.Linear`, no activation
3. **modal_emb is NOT propagated** through any graph — goes directly to fusion
4. **Bootstrap CL is a V-shape (2 edges), NOT a triangle (3 edges)** — no direct connection between item_cf and item_target
5. **item_cf appears ONCE** — it feeds into both fusion (+) and Bootstrap CL from the same single node. Do NOT draw two separate item_cf boxes
6. **NO formulas in the diagram** — only component names and short labels
7. **EMA target encoder** only copies item_embedding (not the full model)
8. **Predictor** box — just the label, no internal formula

## Title
Below the diagram: **"BM3 — Adapted Architecture for Fashion Recommendation (based on Zhou et al., WWW 2023)"**
