# 01 — Prompt: CombiGCN Architecture Diagram

Generate a **high-resolution 4K architecture diagram** (3840×2160 or higher) for our **CombiGCN** recommendation model. Use a clean, academic paper style similar to Fig.1 from the CombiGCN paper. Sharp vector-like rendering, minimal text, professional color palette. This diagram will be used as a **figure only** — formulas and explanations will be written separately below it.

---

## 01_1. Architecture Layout (Bottom-to-Top Data Flow)

```
                             BPR Loss
                                ▲
                                │
                          ŷ = E*ᵤᵀ · E*ᵢ
                                ▲
                    E*ᵤ [553×512]   E*ᵢ [2194×512]
                                ▲
                     mean(layer_0 ... layer_4)
                                ▲
                  ┌─────────────┼─────────────┐
                  │             │             │
             User CF       Item CF       Item Sim
             Branch        Branch        Branch
                  │             │             │
                  │             │       W @ item_emb
                  │             │      [2194 × 2194]
                  │             │             │
               user_cf       item_cf      item_sim
                  │             │             │
                  │             └───── ⊕ ─────┘ (Item Fusion)
                  │                    │
                  │                item_next
                  │                    │
                  └────── concat ◄─────┘
                             │
                          ego_emb
                             ▲
                    E⁰_U [553 × 512] , E⁰_I [2194 × 512]
                        ID Embeddings
                                           similarity_adj (W)
                                           [2194 × 2194]
                                                ▲
                                        Precomputed .npz
                                        (cosine_sim → threshold → normalize)
                                                ▲
                                           Raw Features
                                         (image / text)
```

---

## 01_2. Component-by-Component Specification

### A. Bottom-Left: ID Embeddings
*   **Two embedding blocks:**
    *   `E⁰_U` User ID Embeddings [553 × 512] — grid/matrix icon (light blue).
    *   `E⁰_I` Item ID Embeddings [2194 × 512] — grid/matrix icon (light yellow).
*   **Concatenated input:** `ego_emb = concat(E⁰_U, E⁰_I) [2747 × 512]` feeding into the GCN layers.
*   **Label:** "Item / User ID Embeddings (Xavier init)"

### B. Bottom-Right: Precomputed Similarity Matrix (Data Layer)
*   **Raw Data Source:** `items_features.csv`
*   **Three computation paths (color-coded by sim_type):**
    *   **img_only** (blue): `image vectors → cosine_sim → threshold 0.5 → normalize`
    *   **tfidf** (green): `TF-IDF vectors → cosine_sim → threshold 0.5`
    *   **multimodal** (orange): `BERT_text + Image late fusion (α·text_sim + (1-α)·img_sim) → threshold 0.5 → normalize`
*   **Matrix Destination:** `similarity_adj (W) [2194 × 2194]`
*   **Label:** "Precomputed outside model, cached as .npz (Data Layer)"
*   **Important:** Draw a gray/dashed boundary around this section to show it is computed at the DATA layer, NOT inside the model.

### C. Middle: Propagation Layers — 3 Parallel Branches
Draw **3 vertical branches** running in parallel, with GCN propagation stacked through **4 layers** (layer_1 to layer_4, plus layer_0 as raw embeddings).

*   **Left Branch — "User CF Branch" (light blue background):**
    *   Takes user rows from `interaction_adj @ ego_emb`.
    *   Shows user nodes updated from item interactions.
*   **Center Branch — "Item CF Branch" (light yellow background):**
    *   Takes item rows from `interaction_adj @ ego_emb`.
    *   Shows item representations updated from user interactions.
*   **Right Branch — "Item Similarity Branch" (light pink background):**
    *   Computes `W @ item_emb` where `W` is the precomputed similarity matrix.
    *   **sim_type selector** (blue/green/orange) enters this branch to swap `W`.

*   **Layer-wise Fusion & Recurrent Loop (KEY FEATURE):**
    *   At **EVERY** GCN layer propagation, draw a prominent **⊕ symbol** merging the output of **Item CF Branch** and **Item Sim Branch**:
        $$item\_next = item\_cf + item\_sim$$
    *   Then, draw a **concat** node merging the updated `user_cf` and the fused `item_next`:
        $$ego\_emb = concat(user\_cf, item\_next)$$
    *   Show an arrow feeding this `ego_emb` back as the input to the next layer.
    *   **This is CombiGCN's defining feature** — fusion at every layer, not just at the end.

*   **Layer Aggregation:**
    *   After 4 propagation layers: `final = mean(layer_0, layer_1, layer_2, layer_3, layer_4)`
    *   Output is split into: `E*ᵤ [553 × 512]` and `E*ᵢ [2194 × 512]`.

### D. Top: Prediction & Loss
*   **Dot Product:** $\hat{y}_{ui} = E^*_u \cdot E^*_i$
*   **Target Box:** **"BPR Loss"**
*   **Label:** "No contrastive loss (unlike BM3/FREEDOM)"

---

## 01_3. Visual Style Guidelines (Avoiding Noise)

*   **Colors:**
    *   User CF Branch: light blue background.
    *   Item CF Branch: light yellow background.
    *   Item Sim Branch: light pink background.
    *   Fusion ⊕: Bold white circle with black border.
    *   Precomputed Data Block: Gray dashed border (indicating external computation).
    *   sim_type paths: img_only = blue, tfidf = green, multimodal = orange.
*   **Node representations:**
    *   User nodes: blue circles.
    *   Item nodes: yellow/gold circles.
    *   Embeddings: Grid/matrix icons (dotted rectangles).
*   **Key visual emphasis:**
    *   The **⊕ fusion at every layer** should be the most prominent feature.
    *   The **3 parallel branches** should be clearly separated using the background color blocks.
    *   Show the **feedback loop** where the fused output feeds back as input to the next layer.
*   **General:**
    *   Background: White.
    *   **CRITICAL: NO complex formulas, code snippets, or small dense text in the diagram. Keep all text readable at a glance.** Formulas and detailed math are explained below the figure.

---

## 01_4. Key Points for Accurate Generation

1.  **Data Layer Processing:** Matrix $W$ is built outside the GCN model (Data layer), not learned during training.
2.  **Every-Layer Fusion:** Fusion happens at every propagation layer, which differs from late-fusion models like BM3 or FREEDOM.
3.  **No Contrastive Loss:** The training objective is solely BPR Loss + L2 Regularization.
4.  **No Modal Projectors:** Raw features are processed into the static similarity matrix before training begins.
5.  **3 Similarity Configurations:** img_only, tfidf, multimodal (excluding the "none" / LightGCN baseline in this diagram).

---
