# Data Pipeline

## 1. Input Files

Tat ca nam trong 1 thu muc data (vd: `clip_10k_sample/`):

```
clip_10k_sample/
├── train.txt               # User-item interactions (training)
├── test.txt                # User-item interactions (test)
├── items_features.csv      # Item features (text + image embeddings)
├── social_trust.txt        # (Optional) User-user social trust
└── s_*.npz                 # (Auto-generated) Cached adjacency matrices
```

### 1.1 train.txt / test.txt

```
uid item1 item2 item3 ...
0 15 42 7 88
1 3 56
2 10 23 45 67 89
```

- Moi dong: user_id + danh sach item_ids da tuong tac
- Split 80/20 per user (do preprocess_vcr tao)

### 1.2 items_features.csv

```csv
item_id,feature1,feature2,feature3
0,"Summer Dress casual beach","Light fabric...", "[0.123, -0.456, ...]"
1,"Winter Coat formal","Warm wool...", "[0.789, 0.012, ...]"
```

| Column | Noi dung | Dung cho |
|---|---|---|
| `feature1` | name + outfit_tags | TF-IDF, BERT |
| `feature2` | description | BERT |
| `feature3` | Image embedding vector (string) | Image similarity |

**Luu y**: `feature3` co the la:
- **MobileNetV2**: 768-D (sau PCA tu 1280)
- **CLIP**: 512-D (khong can PCA)
- parse_vector_string() ho tro ca space va comma delimiter

## 2. Data Class — Khoi tao

```python
data = Data(path="clip_10k_sample/", batch_size=1024)
```

### 2.1 Doc train.txt + test.txt

```python
self.train_items = {uid: [item_ids]}   # dict
self.test_set    = {uid: [item_ids]}   # dict
self.exist_users = [uid1, uid2, ...]   # list users co trong train
self.n_users, self.n_items             # int (max_id + 1)
self.n_train, self.n_test              # int (tong so interactions)
```

### 2.2 Tao Interaction Matrix R

```python
self.R = sparse_matrix(n_users, n_items)  # R[u,i] = 1.0 neu user u tuong tac item i
```

### 2.3 Co-occurrence Matrices

```python
self.U = R @ R^T    # (n_users, n_users) — user-user co-interaction
self.I = R^T @ R    # (n_items, n_items) — item-item co-interaction
```

## 3. Adjacency Matrices — 8 loai

`get_norm_adj_mat()` tra ve 8-tuple, moi matrix da normalized `D^{-0.5} A D^{-0.5}`:

| Index | Ten | Shape | Mo ta |
|---|---|---|---|
| 0 | interaction | (n_u+n_i, n_u+n_i) | Bipartite user-item |
| 1 | social | (n_u, n_u) | User-user social trust |
| 2 | similar_users | (n_u, n_u) | User similarity (co-interaction) |
| 3 | tfidf_item | (n_i, n_i) | TF-IDF text similarity > 0.5 |
| 4 | bert_item | (n_i, n_i) | BERT(feature2) * TF-IDF(feature1) |
| 5 | full_bert_item | (n_i, n_i) | BERT(feature1+feature2) |
| 6 | multimodal | (n_i, n_i) | alpha*text_sim + (1-alpha)*img_sim |
| 7 | img_only | (n_i, n_i) | Image embedding cosine sim > 0.5 |

### 3.1 Caching

Moi matrix duoc luu thanh file `.npz` sau lan dau build:

```
clip_10k_sample/
├── s_interaction_adj_mat.npz
├── s_social_adj_mat.npz
├── s_similar_users_adj_mat.npz
├── s_tfidf_item_similarity_adj_mat.npz
├── s_bert_item_similarity_adj_mat.npz
├── s_full_bert_item_similarity_adj_mat.npz
├── s_multimodal_similarity_adj_mat.npz
└── s_img_similarity_adj_mat.npz
```

Lan chay tiep theo se load tu cache (nhanh hon nhieu).

**Luu y**: Neu doi data, can xoa cac file `s_*.npz` de rebuild.

### 3.2 sim_type → Matrix Index

```python
sim_map = {
    "tfidf":      matrices[3],
    "bert":       matrices[4],
    "full_bert":  matrices[5],
    "multimodal": matrices[6],
    "img_only":   matrices[7],
}
# "none" → similarity_adj = None
```

## 4. Similarity Matrix Construction

### 4.1 TF-IDF (`create_tfidf_similarity_matrix`)

```
1. Ket hop feature1 + feature2 → combined text
2. TfidfVectorizer → tfidf_matrix
3. cosine_similarity(tfidf, tfidf) → sim_matrix
4. sim_matrix[sim < 0.5] = 0  (threshold)
5. → sparse matrix
```

### 4.2 Image (`create_img_similarity_matrix`)

```
1. Parse feature3 string → numpy arrays
2. cosine_similarity(image_emb, image_emb) → sim_matrix
3. sim_matrix[sim < 0.5] = 0
4. → sparse matrix
```

### 4.3 Multimodal (`create_multimodal_similarity_matrix`)

4 phuong phap fusion:

| Method | Cach ket hop |
|---|---|
| `late_fusion` | alpha * text_sim + (1-alpha) * img_sim |
| `aggregation` | cosine_sim((text + img) / 2) |
| `pca` | PCA(concat(text, img)) → cosine_sim |
| `attention` | MultiHeadAttention + WeightedAttention → cosine_sim |

Mac dinh: `late_fusion` voi `alpha=0.5`

## 5. Symmetric Normalization

```python
def normalized_sym(adj):
    # D^{-0.5} A D^{-0.5}
    rowsum = adj.sum(1)
    d_inv_sqrt = rowsum^{-0.5}
    d_inv_sqrt[inf] = 0
    D = diag(d_inv_sqrt)
    return D @ adj @ D
```

Muc dich: can bang anh huong cua nodes co nhieu connections vs it connections.

## 6. Convert sang PyG format

```python
from model import scipy_to_sparse_tensor

# scipy.sparse → torch_sparse.SparseTensor
interaction_adj = scipy_to_sparse_tensor(matrices[0], device=device)
similarity_adj  = scipy_to_sparse_tensor(matrices[7], device=device)
```

`SparseTensor` la format dung cho `torch_sparse.matmul()` trong model.

## 7. BPR Sampling

```python
users, pos_items, neg_items = data.sample()
```

- `users`: random `batch_size` users tu `exist_users`
- `pos_items`: voi moi user, random 1 item tu `train_items[user]`
- `neg_items`: voi moi user, random 1 item KHONG co trong `train_items[user]`
- Negative sampling: uniform random over all items
