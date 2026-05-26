# 04 — FREEDOM: Workflow chi tiet

## Tong quan

FREEDOM (Freezing and Denoising Graph Structures for Multimodal Recommendation, ACM MM 2023) tach biet 2 luong thong tin: **CF propagation** tren interaction graph va **content propagation** tren **frozen kNN item-item graph** duoc xay tu modal features. Su dung **InfoNCE contrastive loss** de align 2 views.

**File:** `models/freedom.py`

## Kien truc

```
Input: interaction_adj  [n_users+n_items, n_users+n_items]
       image_feats      [n_items, img_dim]
       text_feats       [n_items, txt_dim]

┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌──────── CF Branch (LightGCN) ──────────────────────────────┐ │
│  │ user_emb, item_emb → ego_emb                               │ │
│  │ for L layers: ego = interaction_adj @ ego                   │ │
│  │ user_cf, item_cf = mean(all layers)                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────── Content Branch (Frozen kNN Graph) ─────────────────┐ │
│  │ 1. Modal projection:                                        │ │
│  │    modal_emb = fusion(proj_img(img), proj_txt(txt))         │ │
│  │                                                             │ │
│  │ 2. Propagation tren frozen item-item graph:                 │ │
│  │    for L layers: modal_emb = item_graph @ modal_emb         │ │
│  │    item_content = mean(all layers)                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  item_final = item_cf + item_content        ← FUSION            │
│                                                                  │
│  Contrastive: InfoNCE(item_cf, item_content) ← ALIGNMENT        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Frozen kNN Item Graph — diem doc dao nhat

FREEDOM xay dung item-item graph **1 lan duy nhat tai init** roi **dong bang (freeze)** — khong update trong qua trinh training.

### Quy trinh xay dung (`build_knn_item_graph`):

```
1. Chon features theo sim_type:
   - img_only:  feats = normalize(image_feats)
   - tfidf:     feats = normalize(text_feats)
   - multimodal/mm_attn: feats = normalize(concat(norm_img, norm_txt))

2. Tinh cosine similarity (batch 256 de tranh OOM)

3. Top-K neighbors (k=10 mac dinh):
   - Moi item → lay k neighbors co similarity cao nhat
   - Fill diagonal = -1 (loai bo self-loop)

4. Symmetrize: them canh nguoc (A → A + A^T)

5. Row-normalize: val / degree[row]

6. Luu dang SparseTensor (register_buffer → di theo model.to(device))
```

> **Tai sao freeze?** Paper cho rang graph structure nen on dinh — cap nhat lien tuc se dan den noisy gradients va overfitting. Freeze giup denoise.

## Loss function (3 thanh phan)

```
L = L_bpr + L_reg + lambda_cl * L_infonce

1. L_bpr = softplus(-(pos_score - neg_score)).mean()

2. L_reg = decay * (||u||^2 + ||pos||^2 + ||neg||^2) / batch

3. L_infonce = CrossEntropy( sim(item_cf, item_content) / tau , labels )
   - sim = cosine similarity matrix [batch, batch]
   - tau = temperature (0.2)
   - labels = arange(batch)  → diagonal la positive pairs
```

> **Khac BM3:** InfoNCE CAN negative pairs (off-diagonal elements trong sim matrix la negatives). BM3 dung bootstrap loss KHONG can negatives.

## Forward pass chi tiet

```python
# 1. CF propagation (LightGCN)
user_cf, item_cf = cf_propagate(interaction_adj)

# 2. Content propagation (frozen kNN graph)
item_content = content_propagate()
#   = mean(modal_emb, item_graph @ modal_emb, item_graph^2 @ modal_emb, ...)

# 3. Final item embedding
item_final = item_cf + item_content

# 4. BPR loss
pos_scores = user_cf[users] · item_final[pos_items]
neg_scores = user_cf[users] · item_final[neg_items]
bpr = softplus(-(pos - neg)).mean()

# 5. InfoNCE contrastive loss
cl = infonce(item_cf[pos_items], item_content[pos_items])

# 6. Total
loss = bpr + reg + cl_weight * cl
```

## sim_type variants

| sim_type | kNN graph features | Modal projection |
|---|---|---|
| `img_only` | normalize(image) | proj_img(feats) |
| `tfidf` | normalize(text) | proj_txt(feats) |
| `multimodal` | concat(norm_img, norm_txt) | (proj_img + proj_txt) / 2 |
| `multimodal_attention` | concat(norm_img, norm_txt) | Linear(concat(img, txt)) |

> Luu y: `multimodal` va `multimodal_attention` dung **cung kNN graph** (build tu concat features) nhung **khac modal projection** (late fusion vs attention fusion).

## Hyperparameters rieng FREEDOM

```
freedom_knn_k     = 10     # So neighbors trong kNN graph
freedom_cl_weight = 0.1    # Weight cua InfoNCE loss
freedom_cl_temp   = 0.2    # Temperature cho InfoNCE
```

## So sanh FREEDOM vs BM3

| Dac diem | BM3 | FREEDOM |
|---|---|---|
| Item-item graph | Khong co | kNN graph (frozen) |
| Content propagation | Khong (chi projection) | Co (L layers tren kNN graph) |
| Contrastive loss | Bootstrap (no negatives) | InfoNCE (co negatives) |
| Target encoder | EMA (momentum update) | Khong co |
| Predictor head | Co | Khong co |
| Complexity | Trung binh (2x LightGCN) | Cao (LightGCN + kNN propagation) |

## Diem dac biet

1. **Frozen graph** — xay 1 lan, khong gradient → on dinh, it overfit
2. **2 luong propagation doc lap** — CF tren user-item, content tren item-item → sau do cong lai
3. **InfoNCE alignment** — ep CF view va content view gan nhau (positive = cung item, negative = khac item trong batch)
4. **kNN graph phu thuoc sim_type** — graph structure thay doi theo cach ket hop features → anh huong lon den ket qua
