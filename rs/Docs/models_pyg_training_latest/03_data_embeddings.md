# 03 — Data & Embeddings: Input Files, Adjacency Matrices, BPR Sampling

---

## Input Files

Tất cả nằm trong 1 thư mục data (ví dụ: `clip_10k_sample/`):

```
clip_10k_sample/
├── train.txt               # User-item interactions (training)
├── test.txt                # User-item interactions (test)
├── items_features.csv      # Item features (text + image embeddings)
├── social_trust.txt        # (Optional) User-user social trust
└── s_*.npz                 # (Auto-generated) Cached adjacency matrices
```

### train.txt / test.txt

```
uid item1 item2 item3 ...
0 15 42 7 88
1 3 56
2 10 23 45 67 89
```

Mỗi dòng: user_id + danh sách item_ids đã tương tác. Split 80/20 per user theo thời gian.

### items_features.csv

```csv
item_id,feature1,feature2,feature3
0,"Summer Dress casual beach","Light fabric...", "[0.123, -0.456, ...]"
```

| Column | Nội dung | Dùng cho |
|---|---|---|
| `feature1` | name + outfit_tags | TF-IDF, BERT |
| `feature2` | description | BERT |
| `feature3` | Image embedding vector (string) | Image similarity |

`feature3` có thể là:
- **MobileNetV2**: 768-D (sau PCA từ 1280)
- **CLIP**: 512-D (không cần PCA)

`parse_vector_string()` hỗ trợ cả space và comma delimiter.

---

## Hai loại Embedding

| Embedding | Dim | Đặc điểm |
|---|---|---|
| **CLIP** | 512-D | Multimodal-aware, không cần PCA |
| **MobileNetV2** | 1280-D gốc → 768-D sau PCA | Nhẹ hơn, cần PCA trước khi dùng |

Data folder tương ứng:
- CLIP → `clip_10k_sample/`
- MobileNetV2 → `mbnv2_10k_sample/`

---

## Data Class — Khởi tạo

```python
data = Data(path="clip_10k_sample/", batch_size=1024)
```

```python
self.train_items = {uid: [item_ids]}   # dict
self.test_set    = {uid: [item_ids]}   # dict
self.exist_users = [uid1, uid2, ...]   # list users có trong train
self.n_users, self.n_items             # int (max_id + 1)
self.n_train, self.n_test              # int (tổng số interactions)

# Interaction Matrix
self.R = sparse_matrix(n_users, n_items)  # R[u,i] = 1.0

# Co-occurrence Matrices
self.U = R @ R^T    # (n_users, n_users)
self.I = R^T @ R    # (n_items, n_items)
```

---

## Adjacency Matrices — 8 loại

`get_norm_adj_mat()` trả về 8-tuple, mỗi matrix đã normalized `D^{-0.5} A D^{-0.5}`:

| Index | Tên | Shape | Mô tả |
|---|---|---|---|
| 0 | interaction | (n_u+n_i, n_u+n_i) | Bipartite user-item |
| 1 | social | (n_u, n_u) | User-user social trust |
| 2 | similar_users | (n_u, n_u) | User similarity (co-interaction) |
| 3 | tfidf_item | (n_i, n_i) | TF-IDF text similarity > 0.5 |
| 4 | bert_item | (n_i, n_i) | BERT(feature2) * TF-IDF(feature1) |
| 5 | full_bert_item | (n_i, n_i) | BERT(feature1+feature2) |
| 6 | multimodal | (n_i, n_i) | alpha*text_sim + (1-alpha)*img_sim |
| 7 | img_only | (n_i, n_i) | Image embedding cosine sim > 0.5 |

### sim_type → Matrix Index

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

### Caching

Mỗi matrix được lưu thành file `.npz` sau lần đầu build:

```
clip_10k_sample/
├── s_interaction_adj_mat.npz
├── s_social_adj_mat.npz
├── s_similar_users_adj_mat.npz
├── s_tfidf_item_similarity_adj_mat.npz
├── s_bert_item_similarity_adj_mat.npz     ← chậm nhất (BERT inference ~5-15 phút)
├── s_full_bert_item_similarity_adj_mat.npz
├── s_multimodal_similarity_adj_mat.npz
└── s_img_similarity_adj_mat.npz
```

Nếu đổi data, xóa cache để rebuild:
```bash
rm clip_10k_sample/s_*.npz
```

---

## Xây dựng Similarity Matrices

### TF-IDF

```
1. Kết hợp feature1 + feature2 → combined text
2. TfidfVectorizer → tfidf_matrix
3. cosine_similarity(tfidf, tfidf) → sim_matrix
4. sim_matrix[sim < 0.5] = 0
5. → sparse matrix
```

### Image

```
1. Parse feature3 string → numpy arrays
2. cosine_similarity(image_emb, image_emb) → sim_matrix
3. sim_matrix[sim < 0.5] = 0
4. → sparse matrix
```

### Multimodal — 4 phương pháp fusion

| Method | Cách kết hợp |
|---|---|
| `late_fusion` | alpha * text_sim + (1-alpha) * img_sim |
| `aggregation` | cosine_sim((text + img) / 2) |
| `pca` | PCA(concat(text, img)) → cosine_sim |
| `attention` | MultiHeadAttention + WeightedAttention → cosine_sim |

Mặc định: `late_fusion` với `alpha=0.5`

Config trong `.env`:
```
MULTIMODAL_METHOD=late_fusion
MULTIMODAL_PCA_COMPONENTS=256
```

---

## Symmetric Normalization

```python
def normalized_sym(adj):
    # D^{-0.5} A D^{-0.5}
    rowsum = adj.sum(1)
    d_inv_sqrt = rowsum ** -0.5
    d_inv_sqrt[inf] = 0
    D = diag(d_inv_sqrt)
    return D @ adj @ D
```

Mục đích: cân bằng ảnh hưởng của nodes có nhiều connections vs ít connections.

---

## Convert sang PyG format

```python
from model import scipy_to_sparse_tensor

interaction_adj = scipy_to_sparse_tensor(matrices[0], device=device)
similarity_adj  = scipy_to_sparse_tensor(matrices[7], device=device)
```

`SparseTensor` là format dùng cho `torch_sparse.matmul()` trong model.

---

## BPR Sampling

```python
users, pos_items, neg_items = data.sample()
```

- `users`: random `batch_size` users từ `exist_users`
- `pos_items`: với mỗi user, random 1 item từ `train_items[user]`
- `neg_items`: với mỗi user, random 1 item KHÔNG có trong `train_items[user]`
