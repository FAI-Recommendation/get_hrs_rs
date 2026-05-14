# Architecture — CombiGCN

## 1. Tong quan kien truc

CombiGCN mo rong LightGCN bang cach them **dual-graph propagation**: moi layer, item embeddings nhan tin hieu tu **2 graph** thay vi 1.

```
                    ┌─────────────────────────┐
                    │   User Embeddings (E_u)  │
                    │   Item Embeddings (E_i)  │
                    └────────────┬────────────┘
                                 │
                    ego = concat([E_u, E_i])
                                 │
                    ┌────────────▼────────────┐
                    │      GCN Layer 1        │
                    │  interaction + similarity│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      GCN Layer 2        │
                    │  interaction + similarity│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      GCN Layer 3        │
                    │  interaction + similarity│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Mean Pooling (L0..L3)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  user_final, item_final  │
                    └─────────────────────────┘
```

## 2. Hai che do hoat dong

### 2.1 LightGCN thuan (`--sim_type none`)

Tuong duong file goc `hr/LightGCN.py`.

Moi layer chi co 1 phep nhan:

```
ego_emb = norm_adj @ ego_emb
```

- `norm_adj`: bipartite user-item adjacency matrix, shape `(n_users+n_items, n_users+n_items)`
- Da normalize: D^{-0.5} A D^{-0.5}
- User nhan tin hieu tu items da tuong tac, va nguoc lai

### 2.2 CombiGCN (`--sim_type multimodal|img_only|tfidf|bert`)

Tuong duong 3 file goc: `LightGCN_bert_img.py`, `LightGCN_only_img.py`, `LightGCN_tfidf_bert.py`.

Moi layer co **2 phep nhan + 1 phep cong**:

```python
# Buoc 1: Bipartite propagation (giong LightGCN)
interaction_emb = norm_adj @ ego_emb

# Buoc 2: Item similarity propagation (phan mo rong)
item_emb_similar = sim_adj @ item_emb_current

# Buoc 3: Fusion (element-wise sum)
item_next = item_interaction + item_similar
user_next = user_interaction  # user khong doi

# Buoc 4: Ghep lai
ego_emb = concat([user_next, item_next])
```

## 3. Chi tiet tung thanh phan

### 3.1 Embedding Layer

```python
user_embedding = nn.Embedding(n_users, embed_dim)   # (n_users, 64)
item_embedding = nn.Embedding(n_items, embed_dim)    # (n_items, 64)
# Init: Xavier Normal (glorot_normal trong TF)
```

- Khong co weight matrix W giua cac layer (dac trung cua LightGCN)
- Khong co activation function (ReLU, etc.)
- Chi co nhung phep nhan sparse matrix va cong

### 3.2 Interaction Adjacency Matrix

```
Shape: (n_users + n_items) x (n_users + n_items)

         users    items
users  [  0    |   R   ]     R = user-item interaction matrix
items  [  R^T  |   0   ]     R^T = transpose
```

Normalized: `D^{-0.5} A D^{-0.5}` (symmetric normalization)

### 3.3 Similarity Adjacency Matrix

```
Shape: n_items x n_items

Cac loai:
  - img_only:    cosine_similarity(image_embeddings) > 0.5
  - tfidf:       cosine_similarity(tfidf_vectors) > 0.5
  - multimodal:  alpha * text_sim + (1-alpha) * image_sim > 0.5
  - bert:        bert_sim * tfidf_sim (elementwise multiply)
```

Normalized: `D^{-0.5} A D^{-0.5}` (cung cach)

### 3.4 Layer Aggregation

```python
# Thu thap embeddings tu tat ca layers (bao gom layer 0 = raw embedding)
all_embs = [layer_0, layer_1, layer_2, layer_3]

# Mean pooling
final = mean(all_embs)  # (n_users+n_items, embed_dim)

# Tach ra
user_final = final[:n_users]
item_final = final[n_users:]
```

### 3.5 Node Dropout

Khi `node_dropout > 0` (chi trong training):

```python
# Random drop edges tu interaction_adj
mask = random(nnz) > dropout_rate
adj_dropped = adj[mask] / (1 - dropout_rate)  # Inverted dropout
```

## 4. So sanh toan hoc

### LightGCN thuan

```
E^(k+1) = D^{-1/2} A D^{-1/2} E^(k)

E_final = (1/(K+1)) * sum(E^(0), E^(1), ..., E^(K))
```

### CombiGCN

```
E_interaction^(k+1) = D^{-1/2} A D^{-1/2} E^(k)

E_user^(k+1) = E_interaction_user^(k+1)

E_item^(k+1) = E_interaction_item^(k+1) + D_s^{-1/2} S D_s^{-1/2} E_item^(k)

E_final = (1/(K+1)) * sum(E^(0), E^(1), ..., E^(K))
```

Trong do:
- `A`: bipartite interaction adjacency
- `S`: item-item similarity adjacency
- `D, D_s`: degree matrices tuong ung

## 5. File model.py — Class diagram

```
CombiGCN(nn.Module)
├── __init__(n_users, n_items, embed_dim, n_layers, decay, node_dropout)
│   ├── user_embedding: nn.Embedding
│   └── item_embedding: nn.Embedding
│
├── get_embedding(interaction_adj, similarity_adj=None)
│   ├── similarity_adj=None  → LightGCN thuan
│   └── similarity_adj!=None → CombiGCN dual-graph
│
├── forward(interaction_adj, similarity_adj, users, pos_items, neg_items)
│   └── return (loss, mf_loss, reg_loss)
│
├── predict(interaction_adj, similarity_adj, users)
│   └── return scores (n_users_batch, n_items)
│
└── _dropout_sparse(adj, dropout)
    └── return adj with dropped edges

scipy_to_sparse_tensor(sp_mat, device)
└── Convert scipy.sparse → torch_sparse.SparseTensor
```
