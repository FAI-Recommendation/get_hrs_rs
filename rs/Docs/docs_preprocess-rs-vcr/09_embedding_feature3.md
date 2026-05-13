# 09 – Tạo feature3: Image Embedding (mean / weighted / max + PCA)

## Mục tiêu

`feature3` là vector số học đại diện cho **nội dung hình ảnh** của mỗi outfit.  
Được tạo bằng cách load file `.npy` từ `embeddings_10k/`, gộp các embedding, rồi giảm chiều bằng PCA.

---

## Input

| Thứ | Nguồn |
|---|---|
| `embeddings_10k/` | 2223 file `.npy`, mỗi file = vector `(1280,)` của 1 ảnh (MobileNetV2) |
| `fe_df['picture.id']` | ID ảnh để tìm đúng file `.npy` |
| `fe_df['displayOrder']` | Dùng cho weighted fusion |

---

## Bước 1 – Tạo đường dẫn đến file `.npy`

```python
embeddings_dir = r'E:\...\output_10k_sample\embeddings_10k'

def create_embedding_paths(picture_ids):
    # picture.id có dạng "picture.abc123" → lấy phần sau dấu "." làm tên file
    return [os.path.join(embeddings_dir, f"{pic_id.split('.')[1]}.npy")
            for pic_id in picture_ids]

fe_df['embedding_paths'] = fe_df['picture.id'].apply(create_embedding_paths)
```

> **Lưu ý:** Vì dataset 10k đã lọc `displayOrder == 0` (1 ảnh/outfit), mỗi `embedding_paths` chỉ có **1 phần tử**.

---

## Bước 2 – 3 cách gộp embedding

### Cách 1: Mean Fusion (mặc định dùng)

```python
def load_mean_embeddings(paths):
    outfit_embeddings = []
    for path in paths:
        if os.path.isfile(path):
            embedding = np.load(path)
            outfit_embeddings.append(embedding.flatten().astype(np.float32))
    if outfit_embeddings:
        return np.mean(outfit_embeddings, axis=0)  # trung bình các vector
    return None

cbf_f31['feature3'] = cbf_f31['embedding_paths'].apply(load_mean_embeddings)
# Kết quả: vector (1280,) per outfit
```

### Cách 2: Weighted Fusion

```python
def load_weighted_embeddings(paths, display_orders):
    outfit_embeddings = []
    for path in paths:
        if os.path.isfile(path):
            embedding = np.load(path).flatten().astype(np.float32)
            outfit_embeddings.append(embedding)

    if outfit_embeddings:
        n = len(outfit_embeddings)
        weights = [1 / (i + 1) for i in range(n)]  # ảnh 0 có trọng số cao nhất
        total = sum(weights)
        weights = [w / total for w in weights]
        return np.sum([w * emb for w, emb in zip(weights, outfit_embeddings)], axis=0)
    return None

cbf_f32['feature3'] = cbf_f32.apply(
    lambda row: load_weighted_embeddings(row['embedding_paths'], row['displayOrder']),
    axis=1
)
```

### Cách 3: Max Fusion

```python
def load_max_embeddings(paths):
    outfit_embeddings = []
    for path in paths:
        if os.path.isfile(path):
            embedding = np.load(path)
            outfit_embeddings.append(embedding.flatten().astype(np.float32))
    if outfit_embeddings:
        return np.max(outfit_embeddings, axis=0)  # giá trị lớn nhất theo từng chiều
    return None

cbf_f33['feature3'] = cbf_f33['embedding_paths'].apply(load_max_embeddings)
```

---

## So sánh 3 cách gộp

| Cách | Công thức | Ý nghĩa | Khi nào dùng |
|---|---|---|---|
| **Mean** | `avg(emb1, emb2, ...)` | Đặc trưng trung bình của outfit | Baseline, ổn định nhất |
| **Weighted** | `Σ wᵢ·embᵢ` (w giảm theo displayOrder) | Ảnh chính (order=0) quan trọng hơn | Khi outfit có nhiều ảnh |
| **Max** | `max(emb1, emb2, ...)` theo chiều | Giữ đặc trưng nổi bật nhất | Khi muốn capture chi tiết đặc sắc |

> Với dataset 10k (1 ảnh/outfit), **3 cách cho kết quả giống nhau** vì chỉ có 1 embedding.

---

## Bước 3 – Chọn cách gộp và gán vào `fe_df`

```python
fe_df['feature3'] = cbf_f31['feature3']   # ← Mean (mặc định)
# fe_df['feature3'] = cbf_f32['feature3'] # Weighted
# fe_df['feature3'] = cbf_f33['feature3'] # Max
```

---

## Bước 4 – Giảm chiều bằng PCA (1280 → 768)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Stack tất cả vector thành matrix (N, 1280)
image_embeddings = np.vstack([np.array(emb) for emb in fe_df['feature3']])

# Chuẩn hóa trước PCA
image_embeddings_normalized = normalize(image_embeddings, norm='l2')

# Giảm chiều: min(768, n_features, n_samples)
n_features = image_embeddings_normalized.shape[1]  # 1280
n_samples  = image_embeddings_normalized.shape[0]  # 2223
n_components = min(768, n_features, n_samples)     # → 768

pca = PCA(n_components=n_components)
image_embeddings_reduced = pca.fit_transform(image_embeddings_normalized)

# Cập nhật lại feature3
fe_df['feature3'] = list(image_embeddings_reduced)
# Kết quả: vector (768,) per outfit
```

**Tại sao giảm chiều?**
- 1280 → 768 để đồng nhất với chiều BERT embedding (768)
- Giúp ma trận similarity tính nhanh hơn
- Loại bỏ noise trong không gian cao chiều

---

## Output

| Bước | Shape vector | Ghi chú |
|---|---|---|
| Sau load `.npy` | `(1280,)` | Output gốc của MobileNetV2 |
| Sau gộp (mean/weighted/max) | `(1280,)` | 1 vector per outfit |
| Sau PCA | `(768,)` | Vector cuối lưu vào `feature3` |

**Lưu vào `items_features.csv`:**
```python
df_c = fe_df[['item_id', 'feature1', 'feature2', 'feature3']]
df_c.to_csv('items_features.csv', index=False)
```

`feature3` được lưu dạng **string biểu diễn list**: `"[0.109, 0.096, -0.012, ...]"`

---

## Vai trò trong `load_data.py`

```python
# load_data.py parse lại feature3 từ string về numpy:
def parse_vector_string(vector_string):
    vector = vector_string.strip('[]').split()
    return np.array([float(x) for x in vector], dtype=np.float32)

image_embeddings = np.vstack(
    self.items_features['feature3'].apply(parse_vector_string).values
)
# → tính cosine similarity → s_img_similarity_adj_mat.npz
```
