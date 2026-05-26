# 02 — CombiGCN: Workflow chi tiet

## Tong quan

CombiGCN la mo hinh **dual-graph GCN**: ket hop dong thoi **bipartite interaction graph** (user-item) va **item similarity graph** (item-item) trong moi layer truyen tin. Khi `sim_type=none`, no thoai hoa thanh LightGCN thuan.

**File:** `models/combigcn.py`

## Kien truc

```
Input: interaction_adj [n_users+n_items, n_users+n_items]
       similarity_adj  [n_items, n_items]  (hoac None)

┌──────────────────────────────────────────────────┐
│  user_embedding  [n_users, dim]  (Xavier init)   │
│  item_embedding  [n_items, dim]  (Xavier init)   │
│                                                  │
│  ego_emb = concat(user_emb, item_emb)            │
│                                                  │
│  for layer in range(n_layers):                   │
│    interaction_emb = interaction_adj @ ego_emb    │
│                                                  │
│    if similarity_adj is not None:                 │
│      user_next = interaction_emb[:n_users]        │
│      item_cf   = interaction_emb[n_users:]        │
│      item_sim  = similarity_adj @ ego_emb[n_users:]│
│      item_next = item_cf + item_sim     ← FUSION │
│      ego_emb   = concat(user_next, item_next)     │
│    else:                                          │
│      ego_emb = interaction_emb   ← LightGCN      │
│                                                  │
│  final = mean(layer_0, layer_1, ..., layer_K)     │
│  user_emb_final = final[:n_users]                 │
│  item_emb_final = final[n_users:]                 │
└──────────────────────────────────────────────────┘
```

## Loss function

```
L = L_bpr + L_reg

L_bpr = mean( softplus( -(pos_score - neg_score) ) )
      trong do: pos_score = u_emb · pos_item_emb
                neg_score = u_emb · neg_item_emb

L_reg = decay * (||u||^2 + ||pos||^2 + ||neg||^2) / batch_size
```

> CombiGCN **khong co contrastive loss** — chi dung BPR + L2 regularization.

## Similarity matrix — xay dung tai Data layer

CombiGCN la model duy nhat su dung **precomputed similarity matrix** (xay dung 1 lan roi cache NPZ). Cac loai:

| sim_type | Cach xay dung | Index |
|---|---|---|
| `none` | Khong dung (LightGCN thuan) | — |
| `img_only` | cosine_sim(image_embeddings), threshold 0.5 | 7 |
| `tfidf` | cosine_sim(TF-IDF(feature1+feature2)), threshold 0.5 | 3 |
| `multimodal` | alpha * text_sim + (1-alpha) * img_sim (late fusion) | 6 |

### Quy trinh build similarity (trong `load_data.py`):

```
1. Doc items_features.csv
2. Trich features:
   - Text: TF-IDF(feature1 + feature2) hoac BERT embeddings
   - Image: parse feature3 (embedding vector string)
3. Tinh cosine similarity tren GPU (cosine_sim_gpu)
   - Batch 512 de tranh OOM
4. Threshold: sim < 0.5 → 0 (loai bo noise)
5. Normalize: D^-0.5 * A * D^-0.5 (symmetric normalization)
6. Cache: save_npz → load_npz lan sau
```

## Forward pass chi tiet

```python
# train.py, trong training loop:

# 1. BPR Sampling
users, pos_items, neg_items = data.sample()

# 2. Forward (CombiGCN-specific: nhan 2 adj matrices)
loss, mf_loss, reg_loss = model(
    interaction_adj,    # bipartite graph
    similarity_adj,     # item-item similarity (hoac None)
    users_t, pos_t, neg_t,
)

# 3. Backward voi AMP
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Predict

```python
@torch.no_grad()
def predict(interaction_adj, similarity_adj, users):
    user_emb, item_emb = get_embedding(interaction_adj, similarity_adj)
    scores = user_emb[users] @ item_emb.T    # [batch, n_items]
    return scores
```

## Diem dac biet

1. **Dual-graph fusion theo cach don gian nhat** — sum truc tiep `item_cf + item_sim` moi layer, khong can learnable weight
2. **Multimodal fusion xay ra NGOAI model** — tai Data layer (similarity matrix), model chi thay 1 sparse matrix
3. **Node dropout** — random drop edges trong interaction graph de regularize (optional, mac dinh tat)
4. **torch_sparse.SparseTensor** — hieu qua hon PyTorch native sparse, dung `matmul` tu torch_sparse
