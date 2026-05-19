# 02 — Kiến Trúc Models: CombiGCN / BM3 / FREEDOM

---

## So sánh nhanh

| | CombiGCN | BM3 | FREEDOM |
|--|----------|-----|---------|
| Item-item graph | Similarity matrix (cosine/attention) | Không có | Frozen kNN graph từ modal |
| Modal dùng cho | Build similarity graph | Projector + bootstrap CL | Build kNN graph + content propagation |
| Contrastive loss | Không | Bootstrap (online ↔ target) | InfoNCE (CF view ↔ content view) |
| Pure CF mode | ✅ (`sim_type=none`) | ❌ | ❌ |

**sim_type** chung cho cả 3 models: `img_only` / `tfidf` / `multimodal` / `multimodal_attention`

---

## Input/Output chung

Tất cả model đọc từ cùng 1 data folder (ví dụ `../get10k_data/clip_10k_sample`):

| File | Nội dung | Shape |
|------|----------|-------|
| `train.txt` | User-item interactions (training) | `n_interactions` rows |
| `test.txt` | User-item interactions (test) | `n_interactions` rows |
| `image_embeddings.npy` | Image features của items | `(n_items, img_dim)` |
| `text_embeddings.npy` | Text features của items | `(n_items, txt_dim)` |
| `s_interaction_adj_mat.npz` | Normalized user-item graph (cache) | `(n_users+n_items, n_users+n_items)` |

---

## CombiGCN

**Paper:** Dual-graph GCN kết hợp interaction graph + item similarity graph.

### Kiến trúc

```
                    ┌─────────────────────────┐
                    │   User Embeddings (E_u)  │
                    │   Item Embeddings (E_i)  │
                    └────────────┬────────────┘
                                 │
                    ego = concat([E_u, E_i])
                                 │
                    ┌────────────▼────────────┐
                    │      GCN Layer k        │
                    │  interaction + similarity│
                    └────────────┬────────────┘
                                 │  (lặp L lần)
                    ┌────────────▼────────────┐
                    │  Mean Pooling (L0..LK)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  user_final, item_final  │
                    └─────────────────────────┘
```

Mỗi layer CombiGCN (khi sim_type ≠ none):

```python
# Bước 1: Bipartite propagation (giống LightGCN)
interaction_emb = norm_adj @ ego_emb

# Bước 2: Item similarity propagation (phần mở rộng)
item_emb_similar = sim_adj @ item_emb_current

# Bước 3: Fusion (element-wise sum)
item_next = item_interaction + item_similar
user_next = user_interaction  # user không đổi

# Bước 4: Ghép lại
ego_emb = concat([user_next, item_next])
```

### Input/Output

```
INPUT
├── interaction_adj     SparseTensor (n_users+n_items, n_users+n_items)
├── similarity_adj      SparseTensor (n_items, n_items)
├── users               Tensor (batch,)
├── pos_items           Tensor (batch,)
└── neg_items           Tensor (batch,)

OUTPUT (forward)
├── loss        scalar  ← BPR + reg
├── bpr_loss    scalar
└── reg_loss    scalar

OUTPUT (predict)
└── scores      Tensor (batch_users, n_items)
```

### sim_type ảnh hưởng đến similarity_adj

| sim_type | Cách build similarity_adj |
|---|---|
| `none` | similarity_adj = None → LightGCN thuần |
| `img_only` | cosine(image_emb, image_emb) > 0.5 |
| `tfidf` | cosine(tfidf_vec, tfidf_vec) > 0.5 |
| `multimodal` | alpha * text_sim + (1-alpha) * img_sim > 0.5 |
| `multimodal_attention` | MultiHeadAttention → cosine_sim |

### Toán học

LightGCN thuần:
```
E^(k+1) = D^{-1/2} A D^{-1/2} E^(k)
E_final = (1/(K+1)) * sum(E^(0), ..., E^(K))
```

CombiGCN:
```
E_user^(k+1) = D^{-1/2} A D^{-1/2} E^(k)  [phần user]
E_item^(k+1) = D^{-1/2} A D^{-1/2} E^(k)  [phần item interaction]
             + D_s^{-1/2} S D_s^{-1/2} E_item^(k)  [item similarity]
E_final = (1/(K+1)) * sum(E^(0), ..., E^(K))
```

### Class diagram

```
CombiGCN(nn.Module)
├── __init__(n_users, n_items, embed_dim, n_layers, decay, node_dropout)
│   ├── user_embedding: nn.Embedding  (Xavier Normal init)
│   └── item_embedding: nn.Embedding
│
├── get_embedding(interaction_adj, similarity_adj=None)
│   ├── similarity_adj=None  → LightGCN thuần
│   └── similarity_adj!=None → CombiGCN dual-graph
│
├── forward(interaction_adj, similarity_adj, users, pos_items, neg_items)
│   └── return (loss, mf_loss, reg_loss)
│
├── predict(interaction_adj, similarity_adj, users)
│   └── return scores (n_users_batch, n_items)
│
└── _dropout_sparse(adj, dropout)
```

---

## BM3

**Paper:** Bootstrap Latent Representations for Multi-modal Recommendation, WWW 2023

### Kiến trúc

```
user/item ID embeddings
        │
   LightGCN propagation (interaction_adj)
        │
  item_emb_cf ──────────────────────────────────────────┐
        │                                               │
  modal projector(s)                            EMA target encoder
  image_projector / text_projector / attention_fusion   │
        │                                               │
  item_emb_modal ──── Bootstrap CL loss ───── item_emb_target
        │
  item_emb = item_emb_cf + item_emb_modal
```

### Input/Output

```
INPUT
├── interaction_adj     SparseTensor (n_users+n_items, n_users+n_items)
├── image_feats         Tensor (n_items, img_dim)   [nếu sim_type != tfidf]
├── text_feats          Tensor (n_items, txt_dim)   [nếu sim_type != img_only]
├── users               Tensor (batch,)
├── pos_items           Tensor (batch,)
└── neg_items           Tensor (batch,)

OUTPUT (forward)
├── loss        scalar  ← BPR + reg + cl_weight × bootstrap_cl_loss
├── bpr_loss    scalar
└── reg_loss    scalar

OUTPUT (predict)
└── scores      Tensor (batch_users, n_items)
```

### sim_type ảnh hưởng đến modal fusion

| sim_type | Cách fusion |
|---|---|
| `img_only` | `image_projector(image_feats)` |
| `tfidf` | `text_projector(text_feats)` |
| `multimodal` | `(img + txt) / 2` |
| `multimodal_attention` | `Linear(concat(img, txt))` |

---

## FREEDOM

**Paper:** Freezing and Denoising Graph Structures for Multimodal Recommendation, ACM MM 2023

### Kiến trúc

```
modal features (image / text / fused)
        │
  build_knn_item_graph()  ← tính 1 lần lúc __init__, FROZEN
        │
  frozen item-item graph (SparseTensor)
        │
  content propagation ──── item_emb_content
                                  │
user/item ID embeddings           │ InfoNCE loss
        │                         │
  LightGCN propagation ── item_emb_cf
        │
  item_emb = item_emb_cf + item_emb_content
```

### Input/Output

```
INPUT
├── interaction_adj     SparseTensor (n_users+n_items, n_users+n_items)
├── image_feats         Tensor (n_items, img_dim)
├── text_feats          Tensor (n_items, txt_dim)
├── users               Tensor (batch,)
├── pos_items           Tensor (batch,)
└── neg_items           Tensor (batch,)

OUTPUT (forward)
├── loss        scalar  ← BPR + reg + cl_weight × InfoNCE
├── bpr_loss    scalar
└── reg_loss    scalar

OUTPUT (predict)
└── scores      Tensor (batch_users, n_items)
```

### Điểm khác biệt với BM3

| | BM3 | FREEDOM |
|--|-----|---------|
| Modal graph | Không dùng | Build kNN graph 1 lần, FROZEN |
| Modal features | → Projector → cộng vào CF mỗi forward | → Build kNN → propagate → InfoNCE với CF view |
| Contrastive | Bootstrap (online ↔ target encoder) | InfoNCE (CF view ↔ content view) |
