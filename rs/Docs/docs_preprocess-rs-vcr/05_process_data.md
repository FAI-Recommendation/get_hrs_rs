# Hàm process_data() và Tạo dataset_VCR_0.5_42_10.csv

## Mục đích

Sau khi có `dataset_VCR.csv` (toàn bộ giao dịch sạch), bước này áp dụng hàm `process_data()` để:
1. **Random sampling**: Lấy ngẫu nhiên một phần trăm dữ liệu (giảm kích thước dataset).
2. **N-core filtering**: Lọc bỏ các user có quá ít tương tác (ít hơn n lần).
3. **ID Mapping**: Ánh xạ các ID gốc (string dài) sang ID số nguyên liên tục (0, 1, 2, ...).
4. **Lưu kết quả** thành file CSV với tên mã hóa tham số.

---

## Cells liên quan: Cell 60 - 68

---

## Đọc dataset_VCR.csv

```python
df = pd.read_csv("/kaggle/working/dataset_VCR.csv")
df
```
Load lại file vừa tạo ở bước trước để xử lý tiếp.

---

## Hàm process_data()

```python
def process_data(df, random_percent=0.1, n_core=10, random_state=42):
```

### Tham số

| Tham số | Giá trị mặc định | Mô tả |
|---------|-----------------|-------|
| `df` | - | DataFrame đầu vào (dataset_VCR.csv) |
| `random_percent` | `0.1` | Tỷ lệ lấy mẫu ngẫu nhiên (0.0 - 1.0). Ví dụ: 0.5 = lấy 50% |
| `n_core` | `10` | Số tương tác tối thiểu để giữ lại một user |
| `random_state` | `42` | Seed cho random, đảm bảo tái tạo kết quả |

### Chi tiết từng bước trong hàm

#### Bước 1: Random Sampling

```python
np.random.seed(random_state)
n_samples = int(len(df) * random_percent)
sampled_df = df.sample(n=n_samples, random_state=random_state)
```
Lấy ngẫu nhiên `random_percent * 100%` dòng từ DataFrame. Ví dụ với `random_percent=0.5`, lấy 50% tổng số giao dịch.

#### Bước 2: N-core Filtering (chỉ theo user)

```python
user_counts = sampled_df['user_id_original'].value_counts()
valid_users = user_counts[user_counts >= n_core].index
filtered_df = sampled_df[sampled_df['user_id_original'].isin(valid_users)]
```
- Đếm số lần xuất hiện của mỗi `user_id_original`.
- Chỉ giữ lại những user có ít nhất `n_core` giao dịch.
- **Lưu ý**: Code đã comment phần n-core cho item (chỉ áp dụng n-core cho user).

#### Bước 3: ID Mapping

```python
# Mapping cho user
unique_users = filtered_df['user_id_original'].unique()
user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users, start=0)}

# Mapping cho item
unique_items = filtered_df['item_id_original'].unique()
item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items, start=0)}

# Áp dụng mapping
mapped_df = filtered_df.copy()
mapped_df['user_id'] = mapped_df['user_id_original'].map(user_id_map)
mapped_df['item_id'] = mapped_df['item_id_original'].map(item_id_map)
```
- Tạo dictionary ánh xạ từ ID gốc (string) sang ID số nguyên liên tục (bắt đầu từ 0).
- Thêm 2 cột mới `user_id` và `item_id` (số nguyên) vào DataFrame, giữ nguyên 2 cột gốc `user_id_original` và `item_id_original`.

#### Bước 4: Lưu file

```python
mapped_df.to_csv(f'dataset_VCR_{random_percent}_{random_state}_{n_core}.csv', index=False)
```
Tên file được tạo tự động từ tham số, ví dụ: `dataset_VCR_0.5_42_10.csv`.

---

## Gọi hàm với tham số thực tế

```python
process_data(
    df,
    random_percent=0.5,   # Lấy 50% dữ liệu
    n_core=10,            # User phải có ít nhất 10 tương tác
    random_state=42       # Seed cố định
)
```

---

## File đầu ra: dataset_VCR_0.5_42_10.csv

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `user_id_original` | string | ID gốc của người dùng |
| `item_id_original` | string | ID gốc của outfit |
| `time` | int | UNIX timestamp |
| `user_id` | int | ID số nguyên mới của user (bắt đầu từ 0) |
| `item_id` | int | ID số nguyên mới của item (bắt đầu từ 0) |

---

## Đọc lại kết quả

```python
df = pd.read_csv("/kaggle/working/dataset_VCR_0.5_42_10.csv")
```
Sau khi tạo file, notebook đọc lại `dataset_VCR_0.5_42_10.csv` để dùng cho các bước tiếp theo.

---

## Sắp xếp dữ liệu

```python
df = df.sort_values(by=["user_id", "time"]).reset_index(drop=True)
```
Sắp xếp theo `user_id` trước, rồi theo `time` trong mỗi user. Điều này cần thiết cho bước chia train/test theo thứ tự thời gian.

---

## Tóm tắt

```
[dataset_VCR.csv]
        |
        v
process_data(random_percent=0.5, n_core=10, random_state=42)
        |
        +-- Sample 50% dữ liệu ngẫu nhiên
        +-- Lọc user có >= 10 tương tác
        +-- Ánh xạ ID gốc → ID số nguyên
        |
        v
[dataset_VCR_0.5_42_10.csv]
(user_id_original, item_id_original, time, user_id, item_id)
```

---

## Lưu ý

- Tên file output tự động mã hóa tham số: `dataset_VCR_{random_percent}_{random_state}_{n_core}.csv`.
- N-core chỉ áp dụng cho **user**, không áp dụng cho item (phần item đã bị comment trong code).
- ID mapping bắt đầu từ **0** (enumerate với start=0), phù hợp với các framework RS dùng 0-indexed.
- Sau khi chạy xong hàm, cần đọc lại file và sắp xếp theo user + time trước khi dùng các bước tiếp theo.
