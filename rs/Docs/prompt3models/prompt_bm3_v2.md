# 02 — Prompt: Adapted BM3 Architecture Diagram

**Role:** Scientific Illustrator
**Task:** Generate a high-resolution 4K architecture diagram (3840×2160) for the "Adapted BM3" recommendation model.
**Style:** Clean, academic paper figure style. Sharp vector graphics, distinct color-coded branches, minimal text.
**Constraint:** No formulas. Use clear labels. The layout must strictly follow the specified V-shape structure.

---

## 1. Overall Layout (Bottom-to-Top)

The diagram has a clear bottom-up flow, splitting into two main parallel branches that converge at the top in a specific V-shape structure.
1.  **Bottom:** Inputs (ID Embeddings & Raw Features).
2.  **Middle-Left:** CF Branch (Graph-based).
3.  **Middle-Right:** Modal Branch (Projection-based).
4.  **Upper-Middle:** Fusion & Bootstrap CL (The V-Shape).
5.  **Top:** Loss Functions.

---

## 2. Component Specification

### A. Bottom Section: Inputs

*   **Bottom-Left (ID Embeddings):**
    *   Two grid icons labeled "User ID Emb" (blue) and "Item ID Emb" (yellow).
    *   Below the "Item ID Emb", draw a faint/shadowed copy of it with a prominent **🔄 circular arrow icon** and a dashed border. Label this shadowed block "EMA Target Item Emb (Frozen)".
    *   Arrows from the main User and Item ID Embs feed into the "CF Branch".

*   **Bottom-Right (Raw Features):**
    *   Two icons: "Image Features" and "Text Features".
    *   Both feed into a selection panel labeled "sim_type Selector" showing 4 colored buttons: "img_only", "tfidf", "multimodal", "mm_attention".
    *   The output of this selector feeds into a single solid blue block labeled "Linear Projectors (No Activation)".
    *   The output arrow from this block is labeled "modal_emb" and feeds into the "Modal Branch".

### B. Middle Section: Two Parallel Branches

*   **Left Branch (CF Branch - Light Blue Background):**
    *   Label the entire background area "CF Branch (GCN Propagation)".
    *   Show the input embeddings entering a stack of 4 circular layers representing "Stacked GCN Layers".
    *   Next to the stack, draw a small icon of a "User-Item Bipartite Graph" with a note "+ Dropout".
    *   The output at the top of this branch splits into two nodes: a blue "user_cf" node and a large yellow "item_cf" node.

*   **Right Branch (Modal Branch - Light Orange Background):**
    *   Label the entire background area "Modal Branch (Direct Projection)".
    *   The "modal_emb" arrow from the bottom passes straight through this section without any graph layers.
    *   It terminates in a large orange node at the top labeled "modal_emb".
    *   Add a text note: "No Graph Propagation here".

### C. Upper-Middle Section: Fusion & Bootstrap CL (The V-Shape)

This is the most prominent part, set against a **Light Yellow Background**.

*   **The V-Shape Structure:** Arrange three large nodes in a distinct 'V' formation.
    *   **Left Point of V:** The large yellow "item_cf" node (output from the CF Branch).
    *   **Center Point of V:** The large orange "modal_emb" node (output from the Modal Branch).
    *   **Right Point of V:** A large yellow node with a dashed border labeled "Target Item Emb (from EMA)", connected by a long dashed arrow from the bottom "EMA Target" block.
*   **Connections (Edges):**
    *   Draw a thick, two-headed arrow between the "item_cf" node and the "modal_emb" node. Label it "Bootstrap Loss (L_boot)".
    *   Draw another thick, two-headed arrow between the "modal_emb" node and the "Target Item Emb" node. Label it "Bootstrap Loss (L_boot)".
    *   **Crucial:** Do NOT draw a direct line between "item_cf" and "Target Item Emb".
*   **Predictor & Fusion:**
    *   Attached to the side of the left "item_cf" node, draw a small green box labeled "Predictor Head".
    *   Above the "item_cf" and "modal_emb" nodes, draw a large white circle with a bold **(+)** sign. Arrows from both nodes feed into it for fusion.

### D. Top Section: Loss

*   A final large box at the top labeled "Total Loss".
*   Inside, divide it into two compartments: "BPR Loss" and "Bootstrap CL Loss".
*   Arrows from "user_cf" and the Fusion (+) output go into "BPR Loss".
*   A prominent arrow from the entire V-shape structure goes into "Bootstrap CL Loss".