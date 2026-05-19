# 01 — Data Pipeline: Raw VCR → Train/Test

Toàn bộ hành trình dữ liệu từ 3 file CSV gốc (VCR dataset) đến 2 file `train.txt` / `test.txt` sẵn sàng cho model. Mỗi bước đều có dữ liệu bị loại bỏ — tài liệu giải thích bị mất ở đâu, bị mất bao nhiêu, và tại sao.

---

## Bức tranh tổng thể

```
RAW DATA (VCR Dataset)
  user_activity_triplets.csv   ~64k giao dịch, ~1600+ users
  picture_triplets.csv         ~50k ảnh (nhiều ảnh / outfit)
  outfits.csv                  thông tin metadata outfit
          │
          ▼
  BƯỚC 1 — slipt_10k_sample.py
          │  Lọc ảnh chính (displayOrder == 0)
          │  Loại outfit không có ảnh hợp lệ
          │  Sample ~10k giao dịch (user có >= 3 GD)
          │
          ├──▶  user_activity_triplets_10k.csv   ~10.000 giao dịch
          ├──▶  picture_triplets_10k.csv          2.223 ảnh (1/outfit)
          └──▶  outfits_10k.csv                   2.223 outfits
          │
          ▼
  BƯỚC 2 — preprocess_vcr_10k.py
          │  Làm sạch (loại outfit null name, loại GD mất ảnh)
          │  N-Core Filter: giữ user có >= 5 giao dịch
          │  ID Mapping: string ID → số nguyên 0-indexed
          │
          ├──▶  dataset_VCR_1.0_42_5.csv          9.455 GD, 553 users, 2.194 items
          │
          ▼
  BƯỚC 3 — Per-user Temporal Split (80/20)
          │  Với mỗi user: 80% GD đầu → train, 20% GD cuối → test
          │
          ├──▶  train.txt    7.350 interactions  (553 users)
          └──▶  test.txt     2.105 interactions  (553 users)
```

---

## Số liệu qua từng bước

| Bước | Giao dịch | Users | Items (outfits) |
|---|---|---|---|
| Raw data | ~64.000 | ~1.600+ | — |
| Sau sample 10k (min ≥ 3 GD/user) | ~10.000 | — | **2.223** |
| Sau N-Core 5 filter | **9.455** | **553** | **2.194** |
| → train.txt | **7.350** | **553** | — |
| → test.txt | **2.105** | **553** | — |

---

## Tại sao mất dữ liệu ở mỗi bước?

| Bước | Lý do mất | Ví dụ |
|---|---|---|
| Bước 1 — lọc ảnh | Outfit chỉ có ảnh displayOrder ≠ 0 | Outfit chỉ có ảnh chụp lưng |
| Bước 1 — merge | Outfit không có name hợp lệ trong outfits.csv | Outfit null name |
| Bước 1 — sample | User có < 3 GD bị loại khỏi sampling pool | User mới chỉ thuê 1 lần |
| Bước 2 — N-Core | User có < 5 GD trong 10k sample bị loại | User chỉ có 3-4 GD sau sample |
| Bước 2 — N-Core (gián tiếp) | Item chỉ được thuê bởi user bị loại → item cũng biến mất | 2.223 → 2.194 items (-29) |

---

## Vì sao cả train lẫn test đều có đúng 553 users?

```
Cách hiểu SAI:
  553 users → 442 vào train, 111 vào test

Cách code thực sự làm:
  Mỗi user được chia độc lập theo thời gian:
    User A (10 GD): GD 1-8 → train   |  GD 9-10 → test
    User B (5 GD):  GD 1-4 → train   |  GD 5    → test
    User C (20 GD): GD 1-16 → train  |  GD 17-20 → test
```

Kết quả: **mỗi user đều xuất hiện trong cả train VÀ test** (per-user temporal split, chuẩn trong RS).

---

## BƯỚC 1 — `slipt_10k_sample.py`

### Bước 1.1 — Lọc ảnh chính (`pick_main_picture_rows`)

```python
main_pictures = pictures[pictures["displayOrder"] == 0].copy()
main_pictures = main_pictures.drop_duplicates(subset=["outfit.id"], keep="first")
```

| displayOrder | Nội dung | Dùng được? |
|---|---|---|
| **0** | Model mặc toàn thân, góc trực diện | ✅ Form dáng, màu sắc, phong cách tổng thể |
| 1 | Chụp từ phía sau | ❌ Thiếu thông tin phía trước |
| 2 | Cận cảnh chất liệu vải | ❌ Chỉ có texture, không đại diện outfit |
| 3 | Ảnh mác / phụ kiện | ❌ Sai hoàn toàn |

Nếu lấy sai ảnh, embedding sẽ mang thông tin sai:

```
Ảnh displayOrder=2 (cận vải ren trắng)
  → MobileNetV2 → vector: "texture ren trắng mịn"
  → Cosine similarity cao với: áo thun có vải ren  ❌

Ảnh displayOrder=0 (váy đỏ toàn thân)
  → MobileNetV2 → vector: "váy dài, màu đỏ, formal"
  → Cosine similarity cao với: váy dạ hội khác     ✅
```

### Bước 1.2 — Merge loại outfit không hợp lệ

```python
merged = interactions.merge(pictures_main, on="outfit.id", how="inner")
merged = merged.merge(outfits, left_on="outfit.id", right_on="id", how="inner")
```

INNER JOIN đảm bảo chỉ giữ giao dịch mà outfit có ảnh chính hợp lệ và name không null.

### Bước 1.3 — Sample ~10k giao dịch theo user

```python
eligible_users = user_counts[user_counts >= 3].index.to_list()
shuffled_users = pd.Series(eligible_users).sample(frac=1.0, random_state=42)

for user_id in shuffled_users:
    selected.append(all_transactions_of_user)
    total += count
    if total >= 10_000:
        break
```

Sample theo user (không phải random dòng) để đảm bảo mỗi user được chọn có đủ lịch sử.

### Đầu ra Bước 1

| File | Rows |
|---|---|
| `user_activity_triplets_10k.csv` | ~10.000 |
| `picture_triplets_10k.csv` | **2.223** (1 ảnh/outfit) |
| `outfits_10k.csv` | **2.223** |

---

## BƯỚC 2 — `preprocess_vcr_10k.py`

### Bước 2.1 — Làm sạch

```python
null_outfit_ids = outfits[outfits["name"].isnull()]["id"]
outfits = outfits.dropna(subset=["name"])
pictures = pictures[~pictures["outfit.id"].isin(null_outfit_ids)]
transactions = transactions[~transactions["outfit.id"].isin(null_outfit_ids)]
transactions = transactions[transactions["outfit.id"].isin(pictures["outfit.id"])]
```

### Bước 2.2 — N-Core Filtering

```python
N_CORE = 5
user_counts = sampled_df["user_id_original"].value_counts()
valid_users = user_counts[user_counts >= N_CORE].index
filtered_df = sampled_df[sampled_df["user_id_original"].isin(valid_users)]
```

Chỉ giữ user có tối thiểu 5 giao dịch. Hiệu ứng dây chuyền: item chỉ được thuê bởi user bị loại → item cũng biến mất (2.223 → 2.194 items, -29).

| Tham số | Giá trị |
|---|---|
| `RANDOM_PERCENT` | `1.0` (lấy 100%) |
| `N_CORE` | `5` |
| `RANDOM_STATE` | `42` |

### Bước 2.3 — ID Mapping

```python
user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
```

Chuyển string ID → integer 0-indexed để build adjacency matrix trong PyTorch Geometric.

### Đầu ra Bước 2

| File | Nội dung |
|---|---|
| `dataset_VCR_1.0_42_5.csv` | 9.455 GD, 553 users, 2.194 items |
| `user_list.txt` | 553 dòng — mapping user |
| `item_list.txt` | 2.194 dòng — mapping item |

---

## BƯỚC 3 — Per-User Temporal Split (80/20)

```python
# Đánh rank theo thời gian cho từng user
df["rank"] = df.groupby("user_id").cumcount() + 1

# Tính ngưỡng train
user_counts["train_threshold"] = (user_counts["total_interactions"] * 0.8).astype(int)

# Chia train/test
for user_id, udf in df.groupby("user_id"):
    th = udf["train_threshold"].iloc[0]
    train_interactions[user_id] = udf[udf["rank"] <= th]["item_id"].tolist()
    test_interactions[user_id]  = udf[udf["rank"] >  th]["item_id"].tolist()
```

Ví dụ User A có 10 GD:

```
rank:   1    2    3    4    5    6    7    8  |  9   10
item: [15] [23] [47] [89][102][134][156][178]|[201][215]
                                              ^
                           train_threshold = 8 (10 * 0.8 = 8)

→ train.txt:  "A 15 23 47 89 102 134 156 178"
→ test.txt:   "A 201 215"
```

Tỉ lệ thực ~77.7% / 22.3% (không phải 80%) do `.astype(int)` làm tròn xuống cho user có GD lẻ.

### Đầu ra Bước 3

| File | Users | Interactions | Avg items/user |
|---|---|---|---|
| `train.txt` | **553** | **7.350** | 13.29 |
| `test.txt` | **553** | **2.105** | 3.81 |

Định dạng:
```
# user_id  item_id1  item_id2  ...
0 15 23 47 89 102 134 156 178
1 5 12 34
```
