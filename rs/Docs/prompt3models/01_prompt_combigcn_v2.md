# 01 — Prompt: CombiGCN Architecture Diagram

**Role:** Scientific Illustrator
**Task:** Generate a high-resolution 4K architecture diagram (3840×2160) for the "CombiGCN" recommendation model.
**Style:** Clean, academic paper figure style. Sharp vector-like graphics, professional muted color palette, minimal text.
**Constraint 1 (Content):** Do not include complex formulas or long explanations. Use simple labels. The diagram flows from bottom to top.
**Constraint 2 (Layout):** Do not include any figure title, caption, or text at the very bottom of the image (e.g., 'Figure 1...'). The final image should contain only the diagram components itself.

---

## 1. Overall Layout (Bottom-to-Top)

The diagram is structured into four main horizontal levels:
1.  **Bottom Level:** Data & Initial Embeddings
2.  **Middle Level:** 4 Stacked Propagation Layers (Layer 1 to Layer 4)
3.  **Aggregation Level:** Multi-layer Aggregation
4.  **Top Level:** Prediction & Loss

---

## 2. Level-by-Level Specification

### A. Bottom Level: Data & Initial Embeddings

*   **Left Side (ID Embeddings):**
    *   Two side-by-side blocks labeled "User ID Embedding (E_U_0)" (blue grid icon) and "Item ID Embedding (E_I_0)" (yellow grid icon).
    *   Arrows from both converge into a central block labeled "Initial Ego Embedding (concat)".
    *   A prominent arrow points upwards from here, labeled "Input to Layer 1".

*   **Right Side (Data Layer - Enclosed in a dashed gray box):**
    *   Label the box "Precomputed Data Layer (.npz)".
    *   Inside, show a source icon labeled "Raw Features (Image/Text)".
    *   Three colored paths originate from it:
        *   **Path 1 (Blue):** "img_only" -> "Cosine Sim & Threshold".
        *   **Path 2 (Green):** "tfidf" -> "Cosine Sim & Threshold".
        *   **Path 3 (Orange):** "multimodal" -> "Late Fusion & Threshold".
    *   All three paths converge into a large matrix icon labeled "Similarity Matrix W (Precomputed)".
    *   A prominent vertical arrow points upwards from this matrix, spanning across all middle layers to serve as input to the "Item Sim Branch" in **EVERY** layer.

### B. Middle Level: 4 Stacked Propagation Layers (CRITICAL FIX)

Draw **4 structurally identical horizontal layer blocks** stacked on top of each other, labeled "Layer 1" (bottom) to "Layer 4" (top).

**EXTREMELY IMPORTANT CONSTRAINT: All 4 layers (Layer 1, 2, 3, AND Layer 4) MUST have the exact same internal processing blocks. Do NOT simplify Layer 4. It must contain the "Graph Conv" and "Matrix Mult" blocks just like Layers 1-3.**

Each of the 4 layer blocks contains 3 parallel vertical branches separated by background color:

1.  **Left Branch (User CF - Light Blue BG):**
    *   Label: "User CF Branch".
    *   Input arrow comes from the layer below.
    *   **MUST include a rectangular process block labeled "Graph Conv (User-Item Graph)".**
    *   The output arrow from this block is labeled "user_cf".

2.  **Center Branch (Item CF - Light Yellow BG):**
    *   Label: "Item CF Branch".
    *   Input arrow comes from the layer below.
    *   **MUST include a rectangular process block labeled "Graph Conv (User-Item Graph)".**
    *   The output arrow from this block is labeled "item_cf".

3.  **Right Branch (Item Sim - Light Pink BG):**
    *   Label: "Item Sim Branch".
    *   Receives input from the layer below AND the "Similarity Matrix W" arrow.
    *   **MUST include a rectangular process block labeled "Matrix Mult (W @ item_emb)".**
    *   The output arrow from this block is labeled "item_sim".

**Layer Fusion (Inside EACH of the 4 layers):**
*   Draw a large, prominent white circle with a bold **(+)** symbol.
*   Arrows **originating from the output of the processing blocks** in the Center Branch ("item_cf") and Right Branch ("item_sim") converge into this (+) circle.
*   The output of the (+) circle is labeled "item_next".
*   Draw a "CONCAT" block that merges the output of the Left Branch processing block ("user_cf") and this "item_next".
*   The output of this CONCAT block is a single arrow pointing UPWARDS to the next layer (or aggregation).

### C. Aggregation & Top Level

*   **Aggregation Block:** Above "Layer 4", draw a wide horizontal block labeled "Multi-layer Mean Aggregation". Five distinct arrows should point into it: one from the "Initial Ego Embedding" and one from the final **CONCAT output** of each of the 4 layers.
*   **Final Embeddings:** The Aggregation Block splits into two final output blocks: "Final User Emb (E_U_star)" (blue) and "Final Item Emb (E_I_star)" (yellow).
*   **Prediction & Loss:**
    *   Arrows from the two final embedding blocks converge into a circle labeled "Dot Product".
    *   The output goes into a final top box labeled "BPR Loss".
    *   Add a small text note next to it: "No Contrastive Loss".