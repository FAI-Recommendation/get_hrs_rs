# Prompt: FREEDOM Architecture Diagram (Adapted Implementation)

Generate a **high-resolution 4K architecture diagram** (3840×2160 or higher) for our **adapted FREEDOM** recommendation model. Use a clean, academic paper style similar to the attached reference figure. Sharp vector-like rendering, minimal text, professional color palette. This diagram will be used as a **figure only** — formulas and explanations will be written separately below it.

**IMPORTANT: This is our ADAPTED implementation, NOT the original paper. Follow this spec exactly.**

## Architecture Layout (bottom-to-top)

```
                         BPR Loss + InfoNCE Loss
                          ↑              ↑
                    ŷ = hᵤᵀ · hᵢ    InfoNCE(item_cf, item_content)
                          ↑              ↑
                         hᵢ = item_cf ⊕ item_content
                               ↑              ↑
                          ┌────┘              └────┐
                          │                        │
                     item_cf                 item_content
                     user_cf                 [2194 × 512]
                          ↑                        ↑
                    [CF Branch]            [Content Branch]
                     LightGCN              kNN Graph Prop
                          ↑                        ↑
                   User-Item Graph          modal_emb ← Linear projectors
                   + Dropout                       ↑
                          ↑                  Raw Features
                     h⁰ᵢ , h⁰ᵤ              (image/text)
                   ID Embeddings
```

## Component-by-Component Specification

### Bottom-Left: ID Embeddings
- Two embedding blocks side by side:
  - `h⁰ᵢ` Item [2194 × 512] — small grid/matrix icon
  - `h⁰ᵤ` User [553 × 512] — small grid/matrix icon
- Label: "Item / User ID Embeddings"
- Style: dotted rectangles with grid pattern (matching paper figure style)

### Bottom-Right: Raw Features + Linear Projection
- Two input sources:
  - `image_feats [2194 × 512]` — small icon
  - `text_feats [2194 × 768]` — small icon
- Arrow down into a box labeled **"Linear Projectors"** (NOT MLP — single layer, no activation)
- Show the projection merging into ONE output: `modal_emb [2194 × 512]`
- **Important:** Draw 4 sim_type variants as a small selector/switch:
  - img_only (blue): proj_img only
  - tfidf (green): proj_txt only
  - multimodal (orange): (proj_img + proj_txt) / 2
  - mm_attention (red): Linear(concat)
- All 4 converge to single `modal_emb`

### Right Branch: Content Branch (light green background)

**Step 1 — kNN Graph (with ❄ snowflake):**
- Draw a small item-item graph with nodes and edges
- Prominent **❄ freeze icon** on the graph
- Label: "Frozen kNN Graph (k=10)"
- Small note: "Built once at init, never updated"

**Step 2 — Content Propagation:**
- `modal_emb` feeds INTO the frozen kNN graph as input
- 4 GCN layers: `emb = kNN_graph @ emb` (each layer has ❄)
- Draw as stacked circle layers (like paper figure)
- Output: `item_content = mean(all layers) [2194 × 512]`
- **KEY: input is modal_emb (projected features), NOT ID embeddings**

### Left Branch: CF Branch (light blue background)

- Input: `ego = concat(h⁰ᵢ, h⁰ᵤ) [2747 × 512]`
- User-Item bipartite graph drawn with 2 layers of nodes
- Include **"Dropout"** label (simple random dropout, edge pruning during training)
- 4 GCN layers: `ego = interaction_adj @ ego`
- Draw as stacked circle layers
- Output split into: `user_cf [553 × 512]` + `item_cf [2194 × 512]`

### Middle: Fusion
- Large **(+)** circle where two branches merge
- Show equation: `hᵢ = item_cf + item_content`
- Arrows from Content Branch (item_content) and CF Branch (item_cf) converging
- `user_cf` and `hᵤ` bypass fusion — goes directly to prediction

### Top: Loss Layer
- Box: **"BPR Loss + InfoNCE Loss"**
- Arrows feeding in:
  - `hᵤ` (user) + `hᵢ` (fused item) → BPR ranking
  - `item_cf` + `item_content` → InfoNCE contrastive alignment
- Keep simple — just the box and arrows, no formula details

## Visual Style

- **Colors:**
  - Content Branch area: light green background
  - CF Branch area: light blue background  
  - Linear Projectors box: solid blue with white text
  - Dropout label: small gray tag
  - Frozen kNN graph: blue ❄ snowflake icon
  - Fusion (+): white circle with black border
  - Loss box: white with black border

- **Nodes:**
  - Item nodes: yellow/gold circles
  - User nodes: blue circles
  - Graph edges: thin gray lines

- **Embeddings:** small grid/matrix icons (dotted rectangles matching paper style)

- **Arrows:** clean, thin, directional

- **Labels:** minimal — only component names and variable symbols

- **Background:** white

- **NO comparison tables, NO detailed formulas** in the diagram

## Key Differences from Original Paper Figure (for accuracy)

The diagram MUST show these differences from the paper:

1. **Linear Projectors** (NOT MLP) — single layer, no activation function
2. **modal_emb as input** to content propagation (NOT ID embeddings h⁰ᵢ)
3. **One fused modal_emb** going into kNN graph (NOT separate visual/text paths)
4. **Dropout** on user-item graph (NOT "Structure Denoising" degree-sensitive pruning)
5. **InfoNCE between item_cf ↔ item_content** (NOT between separate modalities)
6. **Mean pooling all layers** in content propagation (NOT just last layer)
7. **No separate h^v_i / h^t_i** outputs — modal features are fused BEFORE propagation

## Title
Below the diagram: **"FREEDOM — Adapted Architecture for Fashion Recommendation (based on Zhou et al., ACM MM 2023)"**
