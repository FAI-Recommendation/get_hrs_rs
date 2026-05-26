# 03 — BM3: Workflow chi tiet

## Tong quan

BM3 (Bootstrap Latent Representations for Multi-modal Recommendation, WWW 2023) ket hop **LightGCN** cho collaborative filtering voi **modal projectors** de hoc dac trung da phuong thuc. Diem doc dao: su dung **bootstrap contrastive learning** (khong can negative pairs) voi **EMA target encoder** — tuong tu BYOL trong self-supervised learning.

**File:** `models/bm3.py`

## Kien truc

```
Input: interaction_adj  [n_users+n_items, n_users+n_items]
       image_feats      [n_items, img_dim]   (None neu tfidf)
       text_feats       [n_items, txt_dim]   (None neu img_only)

┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  ┌─────────────── CF Branch (LightGCN) ───────────────────┐  │
│  │ user_emb [n_users, dim]                                 │  │
│  │ item_emb [n_items, dim]     ← Online encoder            │  │
│  │ item_emb_target [n_items, dim] ← EMA target (frozen)    │  │
│  │                                                         │  │
│  │ ego = concat(user, item)                                │  │
│  │ for L layers: ego = interaction_adj @ ego               │  │
│  │ final_cf = mean(all layers)                             │  │
│  │ user_cf = final[:n_users]                               │  │
│  │ item_cf = final[n_users:]                               │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────── Modal Branch ───────────────────────────┐  │
│  │ image_projector: Linear(img_dim → dim)                  │  │
│  │ text_projector:  Linear(txt_dim → dim)                  │  │
│  │                                                         │  │
│  │ Fusion (theo sim_type):                                 │  │
│  │   img_only:  modal_emb = proj_img(image_feats)          │  │
│  │   tfidf:     modal_emb = proj_txt(text_feats)           │  │
│  │   multimodal:modal_emb = (proj_img + proj_txt) / 2      │  │
│  │   mm_attn:   modal_emb = Linear(concat(img, txt))       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  item_emb_final = item_cf + modal_emb    ← FUSION            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Loss function (3 thanh phan)

```
L = L_bpr + L_reg + lambda_cl * L_bootstrap

1. L_bpr = mean( softplus( -(pos_score - neg_score) ) )
   - pos_score = user_cf · (item_cf + modal_emb)[pos]
   - neg_score = user_cf · (item_cf + modal_emb)[neg]

2. L_reg = decay * (||u||^2 + ||pos||^2 + ||neg||^2) / batch

3. L_bootstrap = ( L_boot(item_cf, modal_emb)
                 + L_boot(modal_emb, item_target) ) / 2

   Trong do:
   L_boot(online, target) = 2 - 2 * mean( normalize(predictor(online))
                                          · normalize(target.detach()) )
```

## EMA Target Encoder

```
Moi training step:
  target.weight = momentum * target.weight + (1 - momentum) * online.weight

Mac dinh: momentum = 0.995 (cap nhat cham → target on dinh)
```

> **Tai sao can EMA?** Bootstrap CL khong dung negative samples — neu khong co EMA, model se collapse (tat ca embeddings giong nhau). EMA target cung cap 1 "anchor" on dinh de online encoder hoc theo.

## Predictor head

```python
self.predictor = nn.Sequential(
    nn.Linear(dim, dim),
    nn.ReLU(),
    nn.Linear(dim, dim),
)
```

Chi dung trong training (bootstrap loss). Khi predict, khong dung predictor.

## Forward pass chi tiet

```python
# 1. CF propagation (LightGCN)
user_cf, item_cf = propagate(interaction_adj, user_emb, item_emb)

# 2. Modal projection + fusion
modal_emb = _modal_emb()  # theo sim_type

# 3. Final item embedding
item_final = item_cf + modal_emb

# 4. BPR loss
pos_scores = user_cf[users] · item_final[pos_items]
neg_scores = user_cf[users] · item_final[neg_items]
bpr_loss = softplus(-(pos - neg)).mean()

# 5. Bootstrap contrastive loss
_, item_target = propagate(interaction_adj, user_emb, item_emb_target)
cl_loss = (bootstrap(item_cf[pos], modal_emb[pos])
         + bootstrap(modal_emb[pos], item_target[pos])) / 2

# 6. EMA update (AFTER loss computation)
update_target()  # momentum moving average

# 7. Total
loss = bpr_loss + reg_loss + cl_weight * cl_loss
```

## sim_type variants

| sim_type | image_feats | text_feats | Fusion |
|---|---|---|---|
| `img_only` | Co | None | proj_img(feats) |
| `tfidf` | None | Co | proj_txt(feats) |
| `multimodal` | Co | Co | (proj_img + proj_txt) / 2 |
| `multimodal_attention` | Co | Co | Linear(concat(img, txt)) |

## Hyperparameters rieng BM3

```
bm3_momentum   = 0.995    # EMA momentum
bm3_cl_weight  = 0.2      # Weight cua bootstrap contrastive loss
```

## Diem dac biet

1. **Bootstrap CL (khong can negative)** — hoc alignment giua CF view va modal view ma khong can xay dung negative pairs (khac InfoNCE cua FREEDOM)
2. **Dual propagation** — LightGCN chay 2 lan moi forward: 1 cho online encoder, 1 cho target encoder
3. **Modal fusion TAI model layer** — khac CombiGCN (tai Data layer), BM3 dung raw embeddings va hoc projection end-to-end
4. **Predictor asymmetry** — online branch co them predictor head, target branch khong co → ngat gradient collapse
