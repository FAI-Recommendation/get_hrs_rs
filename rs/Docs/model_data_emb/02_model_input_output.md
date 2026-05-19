# 02 – Input / Output của từng Model

## Dữ liệu đầu vào chung

Tất cả model đều đọc từ cùng 1 data folder (ví dụ `../get10k_data/clip_10k_sample`):

| File | Nội dung | Shape |
|------|----------|-------|
| `train.txt` | User-item interactions (training) | `n_interactions` rows |
| `test.txt` | User-item interactions (test) | `n_interactions` rows |
| `image_embeddings.npy` | Image features của items | `(n_items, img_dim)` |
| `text_embeddings.npy` | Text features của items (TF-IDF) | `(n_items, txt_dim)` |
| `s_interaction_adj_mat.npz` | Normalized user-item graph (cache) | `(n_users+n_items, n_users+n_items)` |

---

## CombiGCN

**Paper:** Dual-graph GCN kết hợp interaction graph + item similarity graph

```
INPUT
├── interaction_adj     SparseTensor (n_users+n_items, n_users+n_items)  ← user-item graph
├── similarity_adj      SparseTensor (n_items, n_items)                  ← item-item sim graph
├── image_feats         Tensor (n_items, img_dim)   [nếu sim_type dùng image]
├── text_feats          Tensor (n_items, txt_dim)   [nếu sim_type dùng text]
├── users               Tensor (batch,)
├── pos_items           Tensor (batch,)
└── neg_items           Tensor (batch,)

OUTPUT (forward)
├── loss        scalar  ← BPR + reg
├── bpr_loss    scalar
└── reg_loss    scalar

OUTPUT (predict)
└── scores      Tensor (batch_users, n_items)  ← ranking scores
```

**sim_type ảnh hưởng đến:**
- `none` → similarity_adj = None, chạy như LightGCN thuần
- `img_only` → similarity_adj từ image cosine similarity
- `tfidf` → similarity_adj từ TF-IDF text similarity
- `multimodal` → similarity_adj = avg(img_sim, txt_sim)
- `multimodal_attention` → similarity_adj học trọng số attention

---

## BM3

**Paper:** Bootstrap Latent Representations for Multi-modal Recommendation, WWW 2023

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

**Kiến trúc bên trong:**

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

**sim_type ảnh hưởng đến modal fusion:**
- `img_only` → `image_projector(image_feats)`
- `tfidf` → `text_projector(text_feats)`
- `multimodal` → `(img + txt) / 2`
- `multimodal_attention` → `Linear(concat(img, txt))`

---

## FREEDOM

**Paper:** Freezing and Denoising Graph Structures for Multimodal Recommendation, ACM MM 2023

```
INPUT
├── interaction_adj     SparseTensor (n_users+n_items, n_users+n_items)
├── image_feats         Tensor (n_items, img_dim)   [nếu sim_type != tfidf]
├── text_feats          Tensor (n_items, txt_dim)   [nếu sim_type != img_only]
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

**Kiến trúc bên trong:**

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

**Điểm khác biệt với BM3:**
- BM3: modal features → projector → cộng vào CF embedding mỗi forward
- FREEDOM: modal features → build kNN graph 1 lần → propagate trên frozen graph → InfoNCE với CF view

**sim_type ảnh hưởng đến:**
- Cách build kNN item graph (img_only / tfidf / avg của 2 modality)
- Cách tính modal embedding cho content propagation

---

## So sánh nhanh

| | CombiGCN | BM3 | FREEDOM |
|--|----------|-----|---------|
| Item-item graph | Similarity matrix (cosine/attention) | Không có | Frozen kNN graph từ modal |
| Modal dùng cho | Build similarity graph | Projector + bootstrap CL | Build kNN graph + content propagation |
| Contrastive loss | Không | Bootstrap (online ↔ target) | InfoNCE (CF view ↔ content view) |
| Pure CF mode | ✅ (sim_type=none) | ❌ | ❌ |
