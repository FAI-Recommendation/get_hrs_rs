# Làm Sạch DataFrame outfits

## Mục đích

Sau khi load `outfits.csv`, cần làm sạch DataFrame `outfits` trước khi sử dụng cho các bước tiếp theo. Có hai vấn đề chính cần xử lý:
1. Các dòng có cột `name` bị null → loại bỏ hoàn toàn
2. Các dòng có cột `description` bị null → thay thế bằng giá trị mặc định

---

## Cells liên quan: Cell 12 - 24

---

## Bước 1: Kiểm tra null

```python
outfits.isnull().sum()
print(outfits[['name', 'description']].isnull().sum())
```
Kiểm tra số lượng giá trị null trong từng cột để xác định phạm vi xử lý.

---

## Bước 2: Xác định các outfit.id có name null

```python
outfits_null_name = outfits[outfits['name'].isnull()]
null_outfit_ids = outfits[outfits['name'].isnull()]['id']
```
- `outfits_null_name`: DataFrame chứa các dòng outfit bị thiếu tên.
- `null_outfit_ids`: Series chứa danh sách các `id` của những outfit này. 

**Quan trọng**: `null_outfit_ids` sẽ được sử dụng lại ở bước làm sạch `pictures` và `transactions` để loại bỏ các bản ghi liên quan đến outfit không hợp lệ.

---

## Bước 3: Loại bỏ các dòng có name null

```python
outfits = outfits.dropna(subset=['name'])
```
Lý do: Cột `name` là thông tin bắt buộc để tạo `feature1` (đặc trưng văn bản của item). Nếu không có tên, item không thể được mô tả trong phần features, do đó cần loại bỏ.

---

## Bước 4: Điền giá trị mặc định cho description null

```python
outfits['description'] = outfits['description'].fillna('No description available')
```
Lý do: Cột `description` dùng để tạo `feature2`. Thay vì loại bỏ các outfit không có mô tả (gây mất dữ liệu), thay thế bằng chuỗi `"No description available"` để đảm bảo đầu vào cho mô hình luôn có giá trị.

---

## Bước 5: Reset index

```python
outfits.reset_index(drop=True, inplace=True)
```
Sau khi `dropna`, các index cũ bị gián đoạn. Reset về dãy liên tục bắt đầu từ 0.

---

## Kết quả

Sau bước làm sạch, `outfits` DataFrame:
- **Không còn** dòng nào có `name` là null.
- **Không còn** dòng nào có `description` là null (được điền `"No description available"`).
- Index được reset liên tục từ 0.

### Các cột của outfits sau làm sạch

| Cột | Kiểu | Ghi chú |
|-----|------|---------|
| `id` | string | ID outfit, dùng làm key cho join sau này |
| `name` | string | Không còn null |
| `description` | string | Không còn null (đã fill) |
| `outfit_tags` | list | Danh sách tags (đã parse từ string) |
| `tag_categories` | list | Danh sách nhóm tags (đã parse từ string) |
| `group` | string | Nhóm outfit |

---

## Tóm tắt luồng xử lý

```
outfits (raw)
    |
    +-- Kiểm tra null (name, description)
    |
    +-- Lưu null_outfit_ids (để dùng cho clean pictures/transactions)
    |
    +-- dropna(subset=['name'])  --> loại dòng thiếu name
    |
    +-- fillna(description, 'No description available')
    |
    +-- reset_index()
    |
outfits (cleaned)
```

---

## Lưu ý

- `null_outfit_ids` là một biến toàn cục trong notebook, được sử dụng lại ở bước làm sạch `pictures` (cell 31-32) và `transactions` (cell 41-42).
- Nên xử lý `outfits` **trước** khi xử lý `pictures` và `transactions` vì cần có `null_outfit_ids`.
