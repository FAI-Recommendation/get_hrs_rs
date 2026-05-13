# 05 – Vai Trò của `embeddings_10k/` trong Pipeline

## `embeddings_10k/` là gì?

Thư mục chứa **2223 file `.npy`**, mỗi file là vector đặc trưng hình ảnh shape `(1280,)` của 1 outfit, được tạo bởi MobileNetV2.

---

## Vai trò trong pipeline

```
embeddings_10k/
  abc123.npy  (1280,)
  def456.npy  (1280,)
  ...
       ↓  preprocess-rs-vcr-10k.ipynb
  create_embedding_paths()     → tạo list path .npy theo picture.id
       ↓
  load_mean_embeddings()       → load .npy → gộp → 1 vector/outfit
       ↓
  PCA (1280 → 768)             → giảm chiều
       ↓
  items_features.csv [feature3]
       ↓  load_data.py
  create_img_similarity_matrix()
       ↓
  s_img_similarity_adj_mat.npz
       ↓
  LightGCN (SI matrix — Item-Item similarity graph)
```

---

## Tên file `.npy` được tạo ra sao?

```python
# picture.id có dạng: "picture.abc123def456"
# → tên file .npy = phần sau dấu "."

def create_embedding_paths(picture_ids):
    return [
        os.path.join(embeddings_dir, f"{pic_id.split('.')[1]}.npy")
        for pic_id in picture_ids
    ]

# Ví dụ:
# picture.id = "picture.0000cdba64314d84"
# → file: embeddings_10k/0000cdba64314d84.npy
```

---

## Kết nối với `load_data.py`

```python
# load_data.py parse feature3 từ string về numpy:
def parse_vector_string(vector_string):
    vector = vector_string.strip('[]').split()
    return np.array([float(x) for x in vector], dtype=np.float32)

# Tính similarity:
image_embeddings = np.vstack(
    self.items_features['feature3'].apply(parse_vector_string).values
)
similarity_matrix = cosine_similarity(image_embeddings, image_embeddings)
similarity_matrix[similarity_matrix < 0.5] = 0   # threshold
sparse_matrix = sp.csr_matrix(similarity_matrix)
# → lưu thành s_img_similarity_adj_mat.npz
```

---

## Tóm tắt vai trò

| File | Sinh từ | Dùng cho |
|---|---|---|
| `embeddings_10k/*.npy` | `get_embedding_MBNV2_optimized.py` | Tạo `feature3` |
| `items_features.csv[feature3]` | notebook preprocessing | Input của `load_data.py` |
| `s_img_similarity_adj_mat.npz` | `load_data.py` | SI matrix trong LightGCN |
