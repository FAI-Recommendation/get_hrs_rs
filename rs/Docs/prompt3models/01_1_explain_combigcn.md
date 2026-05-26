# 01_1 — CombiGCN Architecture Explanation

This document provides a detailed, academic explanation of the **CombiGCN** recommendation model architecture, structured for inclusion in research papers or reports.

---

## 01. Architectural Overview

CombiGCN is a recommendation model based on a dual-graph GCN structure. It combines a bipartite user-item interaction graph and an item-item similarity graph to perform joint message passing (Dual-Graph Propagation) at each graph convolutional layer.

The architecture consists of 4 main layers (ordered bottom-to-top):
1.  **Data & Precomputation Layer**
2.  **ID Embedding Layer**
3.  **Propagation & Fusion Layers**
4.  **Prediction & Loss Layer**

---

## 02. Component-by-Component Description

### 02_1. Data & Precomputation Layer (Data Layer)
A key feature of CombiGCN is that the item-item similarity matrix ($W$) is precomputed outside the model training loop, saving significant computational resources. Depending on the `sim_type` configuration, $W$ is constructed using three modalities:
*   **Image-only (img_only):** Extracts visual feature vectors from the raw dataset $\rightarrow$ Computes Cosine similarity $\rightarrow$ Filters noise with a threshold of $0.5$ (similarities $< 0.5$ are set to $0$).
*   **Text-only (tfidf):** Extracts textual TF-IDF features from item attributes $\rightarrow$ Computes Cosine similarity $\rightarrow$ Filters noise with a threshold of $0.5$.
*   **Multimodal (multimodal):** Integrates text similarity (from BERT embeddings) and image similarity using a late fusion formula:
    $$W_{\text{multimodal}} = \alpha \cdot W_{\text{text}} + (1 - \alpha) \cdot W_{\text{image}}$$
    *(where $\alpha = 0.5$ by default, followed by a threshold filter of 0.5).*
*   **Symmetric Normalization:** The thresholded similarity matrix is symmetrically normalized to stabilize graph convolutions:
    $$W = D^{-0.5} \cdot A_{\text{sim}} \cdot D^{-0.5}$$
    where $A_{\text{sim}}$ represents the filtered similarity adjacency matrix and $D$ is its degree matrix. The result is stored as a sparse matrix and loaded directly during model initialization.

### 02_2. ID Embedding Layer
The model initializes user and item ID representations using Xavier initialization:
*   **User Embeddings ($E^0_U$):** Shape $[N_u \times d]$ (e.g., $553 \times 512$).
*   **Item Embeddings ($E^0_I$):** Shape $[N_i \times d]$ (e.g., $2194 \times 512$).
*   **Initial Ego Embeddings:** Concatenated together to form the initial joint embedding matrix:
    $$E^0 = [E^0_U \;\|\; E^0_I] \in \mathbb{R}^{(N_u + N_i) \times d}$$

### 02_3. Propagation & Fusion Layers
Graph convolutions are executed in parallel across 3 branches at each propagation layer $l$:

1.  **User CF Branch:**
    Updates user representations using the bipartite interaction graph:
    $$E^{l+1}_U = \text{Interaction-GCN}(E^l_I)$$
2.  **Item CF Branch:**
    Updates collaborative filtering representations of items from user behaviors:
    $$E^{l+1}_{I, CF} = \text{Interaction-GCN}(E^l_U)$$
3.  **Item Similarity Branch:**
    Propagates item features on the item-item similarity graph using the precomputed matrix $W$:
    $$E^{l+1}_{I, Sim} = W \cdot E^l_I$$

**Layer-wise Fusion & Recurrent Update:**
Unlike late-fusion models, CombiGCN performs an active summation of collaborative filtering and content similarity representations at each propagation step:
$$E^{l+1}_I = E^{l+1}_{I, CF} + E^{l+1}_{I, Sim}$$
The updated user and item representations are then concatenated to form the input for the next GCN layer:
$$E^{l+1} = [E^{l+1}_U \;\|\; E^{l+1}_I]$$
This propagation loop is repeated for $K$ layers (default $K = 4$).

**Multi-layer Aggregation:**
The final representation for user and item nodes is obtained by averaging the representations from all propagation layers, including the initial layer 0:
$$E^*_U = \frac{1}{K+1}\sum_{l=0}^{K} E^l_U, \quad E^*_I = \frac{1}{K+1}\sum_{l=0}^{K} E^l_I$$

### 02_4. Prediction & Loss Layer
*   **Prediction Score ($\hat{y}_{ui}$):** The preference of user $u$ for item $i$ is calculated as the dot product between their final representations:
    $$\hat{y}_{ui} = (E^*_u)^\top \cdot E^*_i$$
*   **Bayesian Personalized Ranking (BPR) Loss:** The model is optimized using BPR loss combined with $L_2$ regularization to prevent overfitting:
    $$\mathcal{L} = \mathcal{L}_{\text{BPR}} + \lambda \mathcal{L}_{\text{reg}}$$
    $$\mathcal{L}_{\text{BPR}} = - \frac{1}{|\mathcal{D}|} \sum_{(u, i, j) \in \mathcal{D}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})$$
    $$\mathcal{L}_{\text{reg}} = \sum_{u} \|E^0_u\|_2^2 + \sum_{i} \|E^0_i\|_2^2 + \sum_{j} \|E^0_j\|_2^2$$
    *(where $i$ is a positive item, $j$ is a sampled negative item, $\lambda$ is the weight decay, and $\sigma$ is the sigmoid function).*
*   **Contrastive Loss:** CombiGCN does not utilize any contrastive loss mechanisms.
