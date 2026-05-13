# Load Dữ liệu Gốc (3 File CSV)

## Mục đích

Bước đầu tiên của pipeline là đọc 3 file CSV gốc từ dataset VCR (Vibrent Clothes Rental) vào các DataFrame để chuẩn bị cho các bước xử lý tiếp theo. Trước khi đọc bằng pandas, mỗi file được mở bằng `open()` để kiểm tra định dạng thực tế.

---

## 1. Load outfits.csv

### Kiểm tra định dạng
```python
with open('/kaggle/input/vibrent-clothes-rental-dataset/outfits.csv', 'r') as f:
    lines = f.readlines()
for i in range(5):
    print(lines[i])
```
Mục đích: kiểm tra delimiter, encoding, và cấu trúc header thực tế của file.

### Đọc vào DataFrame
```python
outfits = pd.read_csv(
    '/kaggle/input/vibrent-clothes-rental-dataset/outfits.csv',
    sep=';',
    on_bad_lines='skip'
)
```
- **`sep=';'`**: File sử dụng dấu chấm phẩy làm delimiter (không phải dấu phẩy thông thường).
- **`on_bad_lines='skip'`**: Bỏ qua các dòng bị lỗi định dạng thay vì báo lỗi.

### Parse cột dạng list
```python
import ast
outfits['outfit_tags'] = outfits['outfit_tags'].apply(ast.literal_eval)
outfits['tag_categories'] = outfits['tag_categories'].apply(ast.literal_eval)
```
Hai cột này được lưu dưới dạng chuỗi biểu diễn Python list (ví dụ: `"['tag1', 'tag2']"`). Dùng `ast.literal_eval` để chuyển về list thực sự.

### Cấu trúc cột outfits

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `id` | string/object | ID duy nhất của outfit |
| `name` | string | Tên bộ trang phục |
| `description` | string | Mô tả bộ trang phục (có thể null) |
| `outfit_tags` | list | Danh sách tags của outfit |
| `tag_categories` | list | Danh sách nhóm tags |
| `group` | string | Nhóm phân loại outfit |

---

## 2. Load picture_triplets.csv

### Kiểm tra định dạng
```python
with open('/kaggle/input/vibrent-clothes-rental-dataset/picture_triplets.csv', 'r') as f:
    lines = f.readlines()
for i in range(5):
    print(lines[i])
```

### Đọc vào DataFrame
```python
pictures = pd.read_csv(
    '/kaggle/input/vibrent-clothes-rental-dataset/picture_triplets.csv',
    sep=';',
    on_bad_lines='skip'
)
```

### Cấu trúc cột pictures

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `outfit.id` | string/object | ID của outfit mà ảnh thuộc về |
| `picture.id` | string/object | ID của ảnh (dạng `outfit_id.picture_id`) |
| `file_name` | string | Tên file ảnh |
| `displayOrder` | int | Thứ tự hiển thị ảnh trong outfit (bắt đầu từ 1) |

Ghi chú: `picture.id` có dạng `"<outfit_id>.<sequential_number>"`, phần sau dấu chấm dùng để tạo đường dẫn embedding.

---

## 3. Load user_activity_triplets.csv

### Kiểm tra định dạng
```python
with open('/kaggle/input/vibrent-clothes-rental-dataset/user_activity_triplets.csv', 'r') as f:
    lines = f.readlines()
for i in range(5):
    print(lines[i])
```

### Đọc vào DataFrame
```python
transactions = pd.read_csv(
    '/kaggle/input/vibrent-clothes-rental-dataset/user_activity_triplets.csv',
    sep=';',
    on_bad_lines='skip'
)
```

### Parse cột datetime
```python
transactions['rentalPeriod.start'] = pd.to_datetime(transactions['rentalPeriod.start'])
transactions['rentalPeriod.end'] = pd.to_datetime(transactions['rentalPeriod.end'])
```
Hai cột thời gian được chuyển từ string sang kiểu `datetime64` của pandas.

### Cấu trúc cột transactions

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `outfit.id` | string/object | ID outfit được thuê |
| `customer.id` | string/object | ID khách hàng thực hiện giao dịch |
| `rentalPeriod.start` | datetime | Ngày bắt đầu thuê |
| `rentalPeriod.end` | datetime | Ngày kết thúc thuê |

---

## Tóm tắt

| DataFrame | File gốc | Delimiter |
|-----------|----------|-----------|
| `outfits` | `outfits.csv` | `;` |
| `pictures` | `picture_triplets.csv` | `;` |
| `transactions` | `user_activity_triplets.csv` | `;` |

Sau bước load, cả 3 DataFrame sẽ được xử lý tiếp ở các bước làm sạch (cell 4-43 trong notebook).
