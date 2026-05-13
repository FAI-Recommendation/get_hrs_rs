# Làm Sạch Transactions và Tạo dataset_VCR.csv

## Mục đích

Bước này làm sạch DataFrame `transactions` (user_activity_triplets) theo hai tầng:
1. Loại bỏ các giao dịch liên quan đến outfit không hợp lệ (không có tên).
2. Đảm bảo tính nhất quán: chỉ giữ lại giao dịch có `outfit.id` tồn tại trong `pictures`.

Sau đó, dữ liệu được chuẩn hóa tên cột và lưu thành file `dataset_VCR.csv` — là file trung gian chính cho toàn bộ pipeline tiếp theo.

---

## Cells liên quan: Cell 34 - 59

---

## Bước 1: Kiểm tra dữ liệu transactions gốc

```python
transactions.info()
transactions.head()
```
Kiểm tra cấu trúc cột và số dòng ban đầu.

---

## Bước 2: Loại bỏ giao dịch có outfit.id không hợp lệ (null name)

```python
# Kiểm tra trước
transactions_filter = transactions[transactions['outfit.id'].isin(null_outfit_ids)]

# Xóa thực sự
transactions_cleaned = transactions[~transactions['outfit.id'].isin(null_outfit_ids)]
```
Tương tự bước làm sạch `pictures`, loại bỏ tất cả giao dịch liên quan đến outfit không có tên.

---

## Bước 3: Kiểm tra tính nhất quán với pictures

### Tìm outfit.id trong transactions nhưng không có trong outfits
```python
outfit_ids = set(outfits['id'])
transaction_outfit_ids = set(transactions['outfit.id'])
foreign_ids = transaction_outfit_ids - outfit_ids

if foreign_ids:
    print("Các outfit.id ngoại lai:", foreign_ids)
else:
    print("Không có outfit.id ngoại lai.")
```

### Tìm outfit.id trong transactions nhưng không có trong pictures
```python
pictures_ids = set(pictures['outfit.id'])
transaction_outfit_ids = set(transactions['outfit.id'])
foreign_ids = transaction_outfit_ids - pictures_ids
```

### Lọc transactions chỉ giữ id có trong pictures
```python
transactions = transactions[transactions['outfit.id'].isin(pictures['outfit.id'])]
```
**Lý do**: Chỉ những outfit có ảnh mới có thể được sử dụng trong RS có đặc trưng hình ảnh (feature3). Loại bỏ giao dịch của outfit không có ảnh.

---

## Bước 4: Kiểm tra lần cuối

```python
# Kiểm tra xem còn outfit.id nào không tồn tại trong outfits không
foreign_ids_df = transactions[~transactions['outfit.id'].isin(outfits['id'])]
if not foreign_ids_df.empty:
    print("Vẫn còn outfit.id không tồn tại trong outfits")
else:
    print("Tất cả outfit.id đều hợp lệ.")

# Kiểm tra xem còn outfit.id nào không tồn tại trong pictures không
foreign_ids_df = transactions[~transactions['outfit.id'].isin(pictures['outfit.id'])]
```

---

## Bước 5: Chuẩn hóa và lưu file dataset_VCR.csv

```python
import time

data_collect = transactions.rename(columns={
    "outfit.id": "item_id_original",
    "customer.id": "user_id_original",
    "rentalPeriod.start": "time"
})

# Chuyển cột time thành UNIX Timestamp (Epoch Time - số giây từ 1/1/1970)
data_collect["time"] = pd.to_datetime(data_collect["time"])
data_collect["time"] = data_collect["time"].apply(lambda x: int(time.mktime(x.timetuple())))

# Chỉ giữ 3 cột cần thiết
data_collect = data_collect[['user_id_original', 'item_id_original', 'time']]

# Lưu ra file CSV
data_collect.to_csv("dataset_VCR.csv", index=False)
```

### Giải thích từng thao tác:
- **Đổi tên cột**: Chuẩn hóa tên từ dạng `outfit.id` sang `item_id_original` (phù hợp với convention của RS framework).
- **Chuyển time sang UNIX timestamp**: Giúp so sánh và sắp xếp thời gian dễ dàng hơn (integer thay vì datetime object).
- **Chỉ giữ 3 cột**: `user_id_original`, `item_id_original`, `time` — đây là định dạng chuẩn của interaction dataset.

---

## File đầu ra: dataset_VCR.csv

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `user_id_original` | string | ID gốc của người dùng (customer.id) |
| `item_id_original` | string | ID gốc của outfit (outfit.id) |
| `time` | int | UNIX timestamp của thời điểm bắt đầu thuê |

---

## Thống kê sau bước này

```python
data_collect.shape                        # Tổng số giao dịch
data_collect["user_id_original"].nunique()  # Số người dùng duy nhất
data_collect["item_id_original"].nunique()  # Số outfit duy nhất
```

---

## Tóm tắt luồng xử lý

```
transactions (raw)
    |
    +-- Loại bỏ outfit.id thuộc null_outfit_ids
    |
    +-- Loại bỏ outfit.id không có trong pictures
    |
    +-- Kiểm tra tính nhất quán với outfits và pictures
    |
    +-- Đổi tên cột + chuyển time sang UNIX timestamp
    |
    +-- Chọn 3 cột: user_id_original, item_id_original, time
    |
    v
[dataset_VCR.csv]
```
