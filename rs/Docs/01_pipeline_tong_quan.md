# 01 – Pipeline Tổng Quan: Từ Raw Data → Model Input

## Mục tiêu

Tài liệu này mô tả toàn bộ luồng xử lý dữ liệu từ 3 file CSV gốc của VCR dataset cho đến khi có đủ file để chạy model LightGCN/NGCF/CombiGCN.

---

## Sơ đồ tổng quan

```
📁 RAW DATA (VCR Dataset)
   user_activity_triplets.csv   (giao dịch thuê đồ)
   picture_triplets.csv         (ảnh của outfit)
   outfits.csv                  (metadata outfit)
         │
         ▼
┌─────────────────────────────────┐
│  BƯỚC 1: slipt_10k_sample.py   │  → Lấy 10k mẫu giao dịch
│  (lọc displayOrder==0,          │     2223 outfits
│   sample theo user)             │     3 file _10k.csv
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  BƯỚC 2: get_embedding_         │  → 2223 ảnh .jpg
│  MBNV2_optimized.py            │     → 2223 file .npy (1280-D)
│  (MobileNetV2 + GPU batch)      │     lưu vào embeddings_10k/
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  BƯỚC 3: preprocess-rs-vcr-    │  → dataset_VCR.csv
│  10k.ipynb                     │     train.txt, test.txt
│  (clean, n-core, ID mapping,   │     user_list.txt, item_list.txt
│   tạo features)                │     items_features.csv
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  BƯỚC 4: load_data.py          │  → s_interaction_adj_mat.npz
│  (tự động khi chạy model)      │     s_img_similarity_adj_mat.npz
│                                 │     s_bert_item_similarity_adj_mat.npz
│                                 │     s_tfidf_item_similarity_adj_mat.npz
│                                 │     s_similar_users_adj_mat.npz
│                                 │     s_social_adj_mat.npz
└─────────────────────────────────┘
         │
         ▼
  🚀 CHẠY MODEL (LightGCN / NGCF / CombiGCN)
```

---

## Chi tiết từng bước

### Bước 1 — Lấy mẫu 10k (`slipt_10k_sample.py`)

| Input | Output |
|---|---|
| `user_activity_triplets.csv` (~64k rows) | `user_activity_triplets_10k.csv` (~10k rows) |
| `picture_triplets.csv` (~50k rows) | `picture_triplets_10k.csv` (2223 rows, displayOrder==0) |
| `outfits.csv` | `outfits_10k.csv` (2223 outfits) |

Xem chi tiết: `02_thuat_toan_split_10k.md`

---

### Bước 2 — Tạo embedding ảnh (`get_embedding_MBNV2_optimized.py`)

| Input | Output |
|---|---|
| `images_2223_main/` (2223 file .jpg) | `embeddings_10k/` (2223 file .npy, shape `(1280,)`) |

Xem chi tiết: `03_embedding_mobilenetv2.md`

---

### Bước 3 — Preprocessing (`preprocess-rs-vcr-10k.ipynb`)

| Input | Output |
|---|---|
| 3 file `_10k.csv` + `embeddings_10k/` | `train.txt`, `test.txt` |
| | `user_list.txt`, `item_list.txt` |
| | `intersection_user.txt`, `image_list.txt` |
| | `items_features.csv` (feature1, feature2, feature3) |

Xem chi tiết: `docs_preprocess-rs-vcr/`

---

### Bước 4 — Build adjacency matrices (`load_data.py`)

Chạy **tự động** lần đầu khi khởi chạy model. Đọc `items_features.csv` + `train.txt` → tính toán → lưu cache `.npz`.

| File `.npz` | Nguồn dữ liệu |
|---|---|
| `s_interaction_adj_mat.npz` | `train.txt` (User-Item graph) |
| `s_img_similarity_adj_mat.npz` | `feature3` (image embedding) |
| `s_bert_item_similarity_adj_mat.npz` | `feature2` (BERT description) |
| `s_tfidf_item_similarity_adj_mat.npz` | `feature1` (TF-IDF name+tags) |
| `s_similar_users_adj_mat.npz` | Jaccard similarity từ `R` matrix |
| `s_social_adj_mat.npz` | `social_trust.txt` (nếu có) |

---

## Thư mục output tổng hợp

```
output_10k_sample/
├── user_activity_triplets_10k.csv
├── picture_triplets_10k.csv
├── outfits_10k.csv
├── images_2223_main/          ← 2223 ảnh .jpg
├── embeddings_10k/            ← 2223 file .npy
├── dataset_VCR.csv
├── dataset_VCR_1.0_42_5.csv
├── user_list.txt
├── item_list.txt
├── image_list.txt
├── intersection_user.txt
├── train.txt
├── test.txt
├── items_features.csv
└── s_*.npz                    ← sinh tự động khi chạy model
```
