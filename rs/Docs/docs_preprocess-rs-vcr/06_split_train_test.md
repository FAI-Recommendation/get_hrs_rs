# Chia Train/Test theo User + Time Rank

## Mục đích

Chia tập dữ liệu tương tác thành tập **train** và **test** theo nguyên tắc:
- **Mỗi user đều xuất hiện trong cả train lẫn test** (không chia ngẫu nhiên theo dòng).
- Chia theo thứ tự thời gian: 80% tương tác đầu tiên (theo thời gian) của mỗi user → train, 20% cuối → test.

Đây là cách chia chuẩn trong Recommender System (leave-last-k-out / temporal split per user).

---

## Cells liên quan: Cell 96 - 113

---

## Bước 1: Thêm cột rank theo từng user

```python
df["rank"] = df.groupby("user_id").cumcount() + 1
```
- `cumcount()` đếm số thứ tự bắt đầu từ 0 trong mỗi nhóm `user_id`.
- `+1` để bắt đầu từ 1.
- Vì df đã được sắp xếp theo `user_id` + `time`, `rank=1` là giao dịch sớm nhất của mỗi user.

Ví dụ: User A có 10 tương tác → rank từ 1 đến 10.

---

## Bước 2: Tính ngưỡng chia train/test cho mỗi user

```python
user_counts = df.groupby("user_id")["rank"].max().reset_index()
user_counts.rename(columns={"rank": "total_interactions"}, inplace=True)
user_counts["train_threshold"] = (user_counts["total_interactions"] * 0.8).astype(int)
```
- `total_interactions`: Tổng số tương tác của mỗi user (= max rank).
- `train_threshold`: Ngưỡng chia. Ví dụ: user có 10 tương tác → `train_threshold = 8`.
  - Rank 1-8 → train
  - Rank 9-10 → test

---

## Bước 3: Gộp ngưỡng vào DataFrame chính

```python
df = df.merge(user_counts, on="user_id", how="inner")
```
Thêm cột `total_interactions` và `train_threshold` vào `df` cho mỗi dòng.

---

## Bước 4: Tạo dictionary train và test

```python
train_interactions = {}
test_interactions = {}

for user_id, user_df in df.groupby('user_id'):
    train_threshold = user_df['train_threshold'].iloc[0]
    
    train_items = user_df[user_df['rank'] <= train_threshold]['item_id'].tolist()
    test_items = user_df[user_df['rank'] > train_threshold]['item_id'].tolist()
    
    train_interactions[user_id] = train_items
    test_interactions[user_id] = test_items
```
- Với mỗi user, lấy danh sách `item_id` của các giao dịch trong train (rank <= threshold) và test (rank > threshold).
- Lưu vào dictionary với key là `user_id`.

---

## Bước 5: Ghi file train.txt và test.txt

```python
# Ghi train.txt
with open('train.txt', 'w') as f:
    for user_id, item_list in train_interactions.items():
        item_list_str = ' '.join(map(str, item_list))
        f.write(f"{user_id} {item_list_str}\n")

# Ghi test.txt
with open('test.txt', 'w') as f:
    for user_id, item_list in test_interactions.items():
        item_list_str = ' '.join(map(str, item_list))
        f.write(f"{user_id} {item_list_str}\n")
```

---

## Định dạng file đầu ra

### train.txt và test.txt

Mỗi dòng gồm: `user_id item_id1 item_id2 item_id3 ...`

Ví dụ:
```
0 15 23 47 89 102 134 156 178
1 5 12 34
2 78 99 203 211 256 312 400 429
```

- Phần tử đầu tiên: `user_id` (số nguyên, 0-indexed).
- Các phần tử tiếp theo: danh sách `item_id` theo thứ tự thời gian.
- Mỗi user chiếm đúng 1 dòng.
- **Tất cả user đều có mặt trong cả hai file** (không có user nào chỉ có train hoặc chỉ có test).

---

## Lý do chọn cách chia này

Notebook có ghi chú: `"=> đang sai chia lại theo hướng, mỗi người dùng, đều có trong train và test (đã fix)"`. Cách chia cũ (train_test_split toàn bộ DataFrame) dẫn đến một số user chỉ xuất hiện trong train hoặc test, gây lỗi khi đánh giá mô hình.

---

## Tóm tắt

```
df (sorted by user_id + time)
        |
        +-- Thêm cột rank (thứ tự tương tác theo thời gian của mỗi user)
        |
        +-- Tính train_threshold = total_interactions * 0.8 (per user)
        |
        +-- Phân chia:
            rank <= threshold → train
            rank > threshold  → test
        |
        +-- Ghi file
        |
        v
[train.txt]  [test.txt]
Format: "user_id item_id1 item_id2 ..."
```

---

## Lưu ý

- Tỷ lệ chia là **80% train / 20% test** (dòng `* 0.8`).
- Chia theo **thứ tự thời gian** (temporal split), không phải ngẫu nhiên.
- Đây là **per-user temporal split**: mỗi user được chia độc lập.
- Các `item_id` trong file là ID số nguyên mới (đã mapping), không phải ID gốc.
