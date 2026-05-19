# 01 — Chi Tiết Từng Bước: Raw → Train/Test

---

## BƯỚC 1 — `slipt_10k_sample.py`: Tạo 10k Sample

### Đầu vào

| File | Kích thước |
|---|---|
| `user_activity_triplets.csv` | ~64.000 giao dịch |
| `picture_triplets.csv` | ~50.000 dòng (4-5 ảnh / outfit) |
| `outfits.csv` | thông tin metadata outfit |

### 5 bước bên trong script

#### Bước 1.1 — Lọc ảnh chính (`pick_main_picture_rows`)

```python
main_pictures = pictures[pictures["displayOrder"] == 0].copy()
main_pictures = main_pictures.drop_duplicates(subset=["outfit.id"], keep="first")
```

Mỗi outfit có 4-5 ảnh với các góc khác nhau:

| displayOrder | Nội dung | Dùng được? |
|---|---|---|
| **0** | Model mặc toàn thân, góc trực diện | ✅ Form dáng, màu sắc, phong cách tổng thể |
| 1 | Chụp từ phía sau | ❌ Thiếu thông tin phía trước |
| 2 | Cận cảnh chất liệu vải | ❌ Chỉ có texture, không đại diện outfit |
| 3 | Ảnh mác / phụ kiện | ❌ Sai hoàn toàn |

**Tại sao chỉ lấy displayOrder=0?**  
Nếu lấy sai ảnh, embedding sẽ mang thông tin sai:

```
Ảnh displayOrder=2 (cận vải ren trắng)
  → MobileNetV2 → vector: "texture ren trắng mịn"
  → Cosine similarity cao với: áo thun có vải ren  ❌  (không liên quan)

Ảnh displayOrder=0 (váy đỏ toàn thân)
  → MobileNetV2 → vector: "váy dài, màu đỏ, formal"
  → Cosine similarity cao với: váy dạ hội khác     ✅  (đúng)
```

Cơ sở học thuật:
- **DeepFashion2** (Ge et al., CVPR 2019): "Frontal View" chứa nhiều Global Features nhất
- **VBPR** (He & McAuley, AAAI 2016): Primary Image giúp embedding hội tụ nhanh và chính xác hơn

**Kết quả:** ~50k ảnh → **N ảnh** (1 ảnh / outfit, chỉ displayOrder=0)

---

#### Bước 1.2 — Merge loại outfit không hợp lệ (`merge_sources`)

```python
merged = interactions.merge(pictures_main, on="outfit.id", how="inner")
merged = merged.merge(outfits, left_on="outfit.id", right_on="id", how="inner")
```

INNER JOIN đảm bảo chỉ giữ giao dịch mà outfit:
- Có ảnh chính hợp lệ (displayOrder=0)
- Có tên (`name` không null) trong outfits.csv

Outfit null name bị loại vì không có thông tin để tạo text feature.

---

#### Bước 1.3 — Sample ~10k giao dịch theo user (`sample_by_users`)

```python
# Chỉ xét user có >= 3 giao dịch
eligible_users = user_counts[user_counts >= 3].index.to_list()

# Shuffle ngẫu nhiên thứ tự user (seed=42)
shuffled_users = pd.Series(eligible_users).sample(frac=1.0, random_state=42)

# Cộng dồn toàn bộ GD của từng user cho đến khi đủ 10.000
for user_id in shuffled_users:
    selected.append(all_transactions_of_user)
    total += count
    if total >= 10_000:
        break
```

**Tại sao sample theo user thay vì random dòng?**

| Cách | Vấn đề |
|---|---|
| Random toàn bộ dòng | User A có thể còn 1 GD → rơi vào test, không có gì để học ở train |
| **Sample theo user** (cách dùng) | Lấy **toàn bộ** GD của mỗi user được chọn → mỗi user có đủ lịch sử |

**User có < 3 giao dịch bị loại** khỏi sampling pool hoàn toàn ở bước này.

---

#### Bước 1.4 — Truy ngược về dòng gốc

```python
sampled_interactions = interactions_orig[interactions_orig["_orig_row_id"].isin(sampled_orig_ids)]
```

Giữ nguyên 100% cột gốc của file interactions gốc (không thêm cột nào từ bước merge).

---

#### Bước 1.5 — Build output (`build_outputs`)

```python
pictures_sampled = pictures_main[pictures_main["outfit.id"].isin(sampled_outfit_ids)]
outfits_sampled  = outfits_all[outfits_all["id"].isin(sampled_outfit_ids)]
```

Assertion kiểm tra tính nhất quán:

```python
assert outfits_out["id"].nunique() == len(pictures_out)
# outfit count == picture count  ✅
```

### Đầu ra Bước 1

| File | Rows | Ghi chú |
|---|---|---|
| `user_activity_triplets_10k.csv` | ~10.000 | Giữ nguyên cột gốc |
| `picture_triplets_10k.csv` | **2.223** | Chỉ displayOrder=0, 1 dòng/outfit |
| `outfits_10k.csv` | **2.223** | Giữ nguyên cột gốc |

---

## BƯỚC 2 — `preprocess_vcr_10k.py`: Làm sạch + N-Core + ID Mapping

### Đầu vào

3 file `_10k.csv` từ Bước 1 + thư mục embeddings ảnh `embeddings_10k/`.

---

### Bước 2.1 — Load và làm sạch

```python
# Loại outfit null name
null_outfit_ids = outfits[outfits["name"].isnull()]["id"]
outfits = outfits.dropna(subset=["name"])

# Loại picture và transaction liên quan đến outfit null name
pictures     = pictures[~pictures["outfit.id"].isin(null_outfit_ids)]
transactions = transactions[~transactions["outfit.id"].isin(null_outfit_ids)]

# Chỉ giữ transaction có outfit tồn tại trong pictures
transactions = transactions[transactions["outfit.id"].isin(pictures["outfit.id"])]
```

Sau bước này, toàn bộ dữ liệu đã nhất quán: mọi giao dịch đều có outfit, outfit đều có ảnh.

---

### Bước 2.2 — N-Core Filtering

```python
N_CORE = 5

user_counts = sampled_df["user_id_original"].value_counts()
valid_users = user_counts[user_counts >= N_CORE].index
filtered_df = sampled_df[sampled_df["user_id_original"].isin(valid_users)]
```

**N-Core Filter nghĩa là gì?**  
Chỉ giữ lại user có **tối thiểu 5 giao dịch** trong 10k sample. User có < 5 GD bị loại hoàn toàn.

**Tại sao cần N-Core?**  
Model (LightGCN, BM3, FREEDOM...) học preference của user từ lịch sử tương tác. Nếu user chỉ có 1-2 GD:
- Train nhận được 1 GD (80%), test nhận 1 GD.
- Với chỉ 1 GD để học, model không đủ signal → embedding của user đó gần như random.
- Đánh giá trên user này không có ý nghĩa thống kê.

```
Ví dụ loại bỏ:
  User X: 4 GD → bị loại  (dưới ngưỡng N_CORE=5)
  User Y: 3 GD → bị loại
  User Z: 5 GD → được giữ  ✅
  User W: 20 GD → được giữ ✅
```

**Hiệu ứng dây chuyền — tại sao items giảm từ 2.223 xuống 2.194?**

```
User X (bị loại) đã thuê outfit A
→ Outfit A CHỈ được thuê bởi User X
→ Outfit A không còn giao dịch nào sau khi lọc
→ Outfit A bị loại khỏi item set

Kết quả: 2.223 outfits → 2.194 outfits  (-29 items)
```

**Tham số thực tế:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `RANDOM_PERCENT` | `1.0` | Lấy 100% 10k sample (không sample thêm) |
| `N_CORE` | `5` | Tối thiểu 5 giao dịch/user |
| `RANDOM_STATE` | `42` | Seed cố định để tái tạo kết quả |

---

### Bước 2.3 — ID Mapping

```python
unique_users = filtered_df["user_id_original"].unique()
user_id_map  = {old_id: new_id for new_id, old_id in enumerate(unique_users)}  # bắt đầu từ 0

unique_items = filtered_df["item_id_original"].unique()
item_id_map  = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
```

Chuyển ID gốc (string dài, ví dụ `"outfit-abc123xyz"`) → số nguyên liên tục 0-indexed (`0, 1, 2, ...`).

Cần thiết vì các RS framework (LightGCN, PyTorch Geometric...) dùng integer index để build ma trận adjacency.

**Mapping được lưu trong:**
- `user_list.txt`: mỗi dòng `user_id_original  user_id`
- `item_list.txt`: mỗi dòng `item_id_original  item_id`

### Đầu ra Bước 2

| File | Nội dung |
|---|---|
| `dataset_VCR_1.0_42_5.csv` | 9.455 GD, 553 users, 2.194 items (sau N-Core) |
| `user_list.txt` | 553 dòng — mapping user |
| `item_list.txt` | 2.194 dòng — mapping item |
| `intersection_user.txt` | Toàn bộ tương tác user-item (chưa chia) |

---

## BƯỚC 3 — Per-User Temporal Split (80/20)

### Tại sao không chia random?

```python
# Cách SAI (đã từng dùng, đã fix):
train, test = train_test_split(df, test_size=0.2)
# → User A có thể chỉ rơi vào train, User B chỉ rơi vào test
# → Không evaluate được hết user, một số user không có lịch sử để học
```

Docs ghi chú lại bugfix này: *"đang sai chia lại theo hướng, mỗi người dùng đều có trong train và test (đã fix)"*

### Cách đúng — per-user temporal split

```python
# Bước 1: Đánh rank theo thời gian cho từng user
df["rank"] = df.groupby("user_id").cumcount() + 1
# rank=1 là GD SỚM NHẤT của user đó

# Bước 2: Tính ngưỡng train cho mỗi user
user_counts["train_threshold"] = (user_counts["total_interactions"] * 0.8).astype(int)

# Bước 3: Chia
for user_id, udf in df.groupby("user_id"):
    th = udf["train_threshold"].iloc[0]
    train_interactions[user_id] = udf[udf["rank"] <= th]["item_id"].tolist()
    test_interactions[user_id]  = udf[udf["rank"] >  th]["item_id"].tolist()
```

### Ví dụ minh họa

```
User A có 10 giao dịch, sắp xếp theo thời gian:

  rank:   1    2    3    4    5    6    7    8  |  9   10
  item: [15] [23] [47] [89] [102][134][156][178]|[201][215]
                                                 ^
                              train_threshold = 8 (10 * 0.8 = 8)

  → train.txt:  "A 15 23 47 89 102 134 156 178"
  → test.txt:   "A 201 215"
```

**Kết quả:** mỗi user xuất hiện trong CẢ HAI file với lịch sử khác nhau theo thời gian.

### Tại sao tỉ lệ thực là 77.7% / 22.3% thay vì đúng 80%?

Do `.astype(int)` làm tròn xuống:

```python
# User có 5 GD:  5 * 0.8 = 4.0  → train_threshold = 4  → 4 train, 1 test  = 80%
# User có 6 GD:  6 * 0.8 = 4.8  → train_threshold = 4  → 4 train, 2 test  = 67%
# User có 7 GD:  7 * 0.8 = 5.6  → train_threshold = 5  → 5 train, 2 test  = 71%
```

Tổng hợp qua 553 users với số GD khác nhau → trung bình ra ~77.7% train.

### Đầu ra Bước 3

| File | Users | Interactions | Avg items/user |
|---|---|---|---|
| `train.txt` | **553** | **7.350** | 13.29 |
| `test.txt` | **553** | **2.105** | 3.81 |

**Định dạng file:**
```
# Mỗi dòng: user_id  item_id1  item_id2  item_id3  ...
0 15 23 47 89 102 134 156 178
1 5 12 34
2 78 99 203 211 256 312
```

---

## Tổng kết số liệu hao hụt

```
~64.000 GD  (raw)
    │
    │  Loại outfit null name
    │  Loại outfit không có ảnh chính
    │  Loại user có < 3 GD  (không đủ để sample)
    ▼
~10.000 GD  → 2.223 outfits / 2.223 ảnh
    │
    │  N-Core filter: loại user có < 5 GD
    │  (Hiệu ứng: 29 outfits mất theo user bị loại)
    ▼
 9.455 GD  →  553 users  →  2.194 items
    │
    │  Per-user temporal split 80/20
    ▼
train.txt: 7.350 GD  (553 users)
test.txt:  2.105 GD  (553 users)
```
