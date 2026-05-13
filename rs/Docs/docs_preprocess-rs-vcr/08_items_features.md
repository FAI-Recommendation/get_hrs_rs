# 08 – Tạo items_features.csv

## Mục tiêu

`items_features.csv` là file **đặc trưng item** — đầu vào quan trọng nhất của `load_data.py`.  
Nó chứa 3 loại đặc trưng của mỗi outfit để `load_data.py` tính các ma trận similarity `.npz`.

---

## Input

| Biến | Nguồn |
|---|---|
| `items_features_df` | Merge giữa `outfits` (name, description, tags) và `picture_triplets` (file_name, picture.id) theo `outfit.id` |

```python
items_features_df = pd.merge(
    item_df,          # item_id, item_id_original
    outfits,          # name, description, outfit_tags, tag_categories
    left_on='item_id_original', right_on='id',
    how='inner'
)
# Merge thêm picture info (file_name, picture.id, displayOrder)
```

---

## Cấu trúc `items_features_df` sau merge

| Cột | Kiểu | Mô tả |
|---|---|---|
| `item_id` | int | ID số liên tiếp của item |
| `item_id_original` | str | outfit.id gốc |
| `name` | str | Tên outfit |
| `description` | str | Mô tả outfit |
| `outfit_tags` | str | Tags của outfit (đã parse từ list → string) |
| `tag_categories` | str | Danh mục tags |
| `file_name` | str | Tên file ảnh chính (displayOrder == 0) |
| `picture.id` | str | ID ảnh |
| `displayOrder` | int | Thứ tự hiển thị ảnh |

---

## Tạo `feature1` và `feature2`

```python
fe_df = items_features_df.copy()

# feature1: name + outfit_tags → đặc trưng ngắn, dùng cho TF-IDF
fe_df['feature1'] = fe_df['name'] + ' ' + fe_df['outfit_tags']

# feature2: description → đặc trưng dài, dùng cho BERT
fe_df['feature2'] = fe_df['description']
```

| Feature | Nội dung | Dùng cho |
|---|---|---|
| `feature1` | `"Yellow Blouse ILAG Tops Spring Summer..."` | `create_tfidf_similarity_matrix()` trong `load_data.py` |
| `feature2` | `"This beautiful blouse features an adjustable neckline..."` | `create_bert_similarity_matrix()` trong `load_data.py` |

---

## Tạo `feature3` (image embedding)

Xem chi tiết tại `09_embedding_feature3.md`.  
Tóm tắt: load các file `.npy` từ `embeddings_10k/`, gộp lại thành 1 vector per outfit, giảm chiều bằng PCA.

```python
fe_df['feature3'] = cbf_f31['feature3']   # dùng mean fusion (mặc định)
# fe_df['feature3'] = cbf_f32['feature3'] # hoặc weighted fusion
# fe_df['feature3'] = cbf_f33['feature3'] # hoặc max fusion
```

---

## Lưu file `items_features.csv`

```python
df_c = fe_df[['item_id', 'feature1', 'feature2', 'feature3']]
df_c.to_csv('items_features.csv', index=False)
```

---

## Output

**File:** `output_10k_sample/items_features.csv`

| Cột | Kiểu | Ví dụ |
|---|---|---|
| `item_id` | int | `0` |
| `feature1` | str | `"Yellow Shell Blouse ILAG Tops Spring..."` |
| `feature2` | str | `"This beautiful blouse features an adjustable neckline..."` |
| `feature3` | str (vector) | `"[0.198, 0.458, 0.012, ...]"` — vector 768 chiều sau PCA |

---

## Vai trò trong pipeline

```
items_features.csv
       ↓  load_data.py đọc
feature1 → create_tfidf_similarity_matrix()  → s_tfidf_item_similarity_adj_mat.npz
feature2 → create_bert_similarity_matrix()   → s_bert_item_similarity_adj_mat.npz
feature3 → create_img_similarity_matrix()    → s_img_similarity_adj_mat.npz
feature2+feature3 → create_multimodal_similarity_matrix() → s_multimodal_similarity_adj_mat.npz
```
