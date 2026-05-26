# 03 — Prompt: Adapted FREEDOM Architecture Diagram

**Role:** Scientific Illustrator
**Task:** Generate a high-resolution 4K architecture diagram (3840×2160) for the "Adapted FREEDOM" fashion recommendation model.
**Style:** Clean, academic paper figure style. Sharp vector graphics, distinct color coding for two main branches.
**Constraint:** Absolutely no complex mathematical formulas ($\mathcal{L}$, $\sum$, etc.). Use descriptive text labels only. The flow is strictly bottom-up.

---

## 1. Overall Layout (Bottom-to-Top)

The diagram is structured with inputs at the bottom, two main parallel processing branches in the middle, a fusion point, and loss functions at the top.
1.  **Bottom:** Inputs (Learnable ID Embeddings & Frozen Raw Features).
2.  **Middle-Left:** CF Branch (Interactive View).
3.  **Middle-Right:** Content Branch (Semantic View).
4.  **Top:** Fusion and Three-Part Loss Layer.

---

## 2. Component Specification

### A. Bottom Section: Inputs

*   **Bottom-Left (Learnable IDs):**
    *   A block labeled "Learnable ID Embeddings (Xavier Init)".
    *   Inside, two grid icons: "User Emb (h_u)" (blue) and "Item Emb (h_i)" (yellow).
    *   An arrow upwards leads to the CF Branch.

*   **Bottom-Right (Frozen Features):**
    *   A block labeled "Frozen Raw Features".
    *   Icons for "Image Feats (CLIP)" and "Text Feats (TF-IDF)".
    *   Both feed into a "Linear Projector" box.
    *   The output arrow is labeled "modal_emb" and leads to the Content Branch.

### B. Middle Section: Two Parallel Branches

*   **Left Branch (CF Branch - Blue Background):**
    *   Label: "CF Branch (Interactive View)".
    *   Show the inputs entering a stack of circles labeled "LightGCN Propagation (4 Layers)".
    *   Include an icon of a "User-Item Graph" with "Edge Dropout".
    *   The output of the stack goes to a "Mean Pooling" block.
    *   Final outputs are two separate nodes: "Final User CF" (blue) and "Final Item CF" (yellow).

*   **Right Branch (Content Branch - Green Background):**
    *   Label: "Content Branch (Semantic View)".
    *   **Crucial Step:** The input "modal_emb" first enters a large process block labeled **"Frozen kNN Graph Construction"**.
    *   Inside this block, show a small flow: "Cosine Sim" -> "Top-K Neighbors" -> "Symmetric Norm". Show a small icon of a sparse item-item graph marked "Frozen (No Gradient)".
    *   The output of this construction block, along with the "modal_emb", enters a stack of circles labeled "GCN on Frozen Graph (4 Layers)".
    *   The output goes to a "Mean Pooling" block.
    *   Final output is one node: "Final Item Content" (green).

### C. Top Section: Fusion & Loss

*   **Fusion:**
    *   Above the branches, draw a central circle with a **(+)** sign.
    *   Arrows from "Final Item CF" (yellow) and "Final Item Content" (green) merge into this (+) circle.
    *   The output is a combined item node labeled "Fused Item Emb".

*   **Loss Layer:**
    *   A large top box labeled "Training Objectives (Loss)". Divide it into 3 compartments:
        1.  **"BPR Ranking Loss":** Receives arrows from "Final User CF" and "Fused Item Emb".
        2.  **"InfoNCE Contrastive Loss":** Receives arrows directly from "Final Item CF" and "Final Item Content" (to align the two views).
        3.  **"L2 Regularization":** Receives arrows all the way from the initial ID Embeddings at the bottom.