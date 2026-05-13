# Tổng quan Notebook: preprocess-rs-vcr.ipynb

## Mục tiêu

Notebook này thực hiện **toàn bộ pipeline tiền xử lý dữ liệu** cho bài toán **Recommender System (RS)** dựa trên dataset thuê quần áo VCR (Vibrent Clothes Rental). Kết quả cuối cùng là tập các file sẵn sàng cho việc huấn luyện mô hình gợi ý sản phẩm (outfit).

---

## Nguồn dữ liệu đầu vào (Input)

Dữ liệu gốc nằm trên Kaggle tại:
```
/kaggle/input/vibrent-clothes-rental-dataset/
```

| File | Mô tả |
|------|-------|
| `outfits.csv` | Thông tin các bộ trang phục (outfit): tên, mô tả, tags, nhóm |
| `picture_triplets.csv` | Thông tin ảnh của từng outfit: picture_id, file_name, thứ tự hiển thị |
| `user_activity_triplets.csv` | Lịch sử thuê đồ của người dùng: customer_id, outfit_id, thời gian thuê |

Ngoài ra, embeddings ảnh (vector đặc trưng) được trích xuất từ mô hình MobileNetV2:
```
/kaggle/input/mobilenet-emb/embeddings_MBNV2_full/
```
Mỗi ảnh được lưu dưới dạng file `.npy` tên `<picture_id>.npy`, kích thước vector 1280 chiều.

---

## Các file đầu ra (Output)

| File | Mô tả |
|------|-------|
| `dataset_VCR.csv` | Dataset tương tác gốc sau làm sạch: user_id_original, item_id_original, time |
| `dataset_VCR_0.5_42_10.csv` | Dataset sau khi lọc n-core và mapping ID (50% sample, seed 42, n_core=10) |
| `train.txt` | Tập huấn luyện: mỗi dòng `user_id item_id1 item_id2 ...` |
| `test.txt` | Tập kiểm tra: mỗi dòng `user_id item_id1 item_id2 ...` |
| `user_list.txt` | Danh sách mapping user: `user_id_original user_id` |
| `item_list.txt` | Danh sách mapping item: `item_id_original item_id` |
| `intersection_user.txt` | Tương tác user-item: mỗi dòng `user_id item_id1 item_id2 ...` (toàn bộ) |
| `image_list.txt` | Danh sách ảnh: `item_id_original item_id file_name_list` |
| `items_features.csv` | Đặc trưng item: item_id, feature1, feature2, feature3 |

---

## Luồng xử lý tổng quát

```
[outfits.csv]          [picture_triplets.csv]    [user_activity_triplets.csv]
      |                         |                              |
      v                         v                              v
 Làm sạch outfits        Lọc outfit.id            Lọc outfit.id + parse time
 (drop null name,        không hợp lệ             không hợp lệ
  fill description)            |                              |
      |                        |                              |
      +------------------------+------------------------------+
                               |
                     Kiểm tra tính nhất quán
                     (transactions chỉ giữ id
                      có trong pictures)
                               |
                               v
                        [dataset_VCR.csv]
                               |
                               v
                    process_data() - n-core filter
                    + random sampling (50%)
                    + ID mapping
                               |
                               v
                   [dataset_VCR_0.5_42_10.csv]
                               |
              +----------------+------------------+
              |                |                  |
              v                v                  v
    Chia train/test    Tạo user_list.txt     Tạo intersection_user.txt
    (80/20 theo rank)  item_list.txt         image_list.txt
              |
              v
      [train.txt, test.txt]
              
              +------ Tạo items_features.csv -------+
              |   feature1: name + outfit_tags       |
              |   feature2: description              |
              |   feature3: image embedding (PCA)    |
              +--------------------------------------+
```

---

## Các bước xử lý chính (tương ứng các file MD)

| Bước | File MD | Nội dung |
|------|---------|----------|
| 1 | `01_load_data.md` | Load 3 file CSV gốc |
| 2 | `02_clean_outfits.md` | Làm sạch outfits |
| 3 | `03_clean_pictures.md` | Làm sạch picture_triplets |
| 4 | `04_clean_transactions.md` | Làm sạch user_activity_triplets, tạo dataset_VCR.csv |
| 5 | `05_process_data.md` | Hàm process_data(), tạo dataset_VCR_0.5_42_10.csv |
| 6 | `06_split_train_test.md` | Chia train/test theo user + time rank |
| 7 | `07_user_item_lists.md` | Tạo các file danh sách |
| 8 | `08_items_features.md` | Tạo items_features.csv |
| 9 | `09_embedding_feature3.md` | Load .npy → mean/weighted/max → PCA → feature3 |

---

## Thư viện sử dụng

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import ast, os, time
import cv2
```

---

## Môi trường chạy

Notebook được chạy trên **Kaggle Notebook** (Linux, GPU). Đường dẫn dữ liệu đều bắt đầu bằng `/kaggle/input/` và `/kaggle/working/`.
