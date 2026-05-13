# Làm Sạch DataFrame pictures (picture_triplets)

## Mục đích

Sau khi load `picture_triplets.csv`, cần loại bỏ tất cả các bản ghi ảnh liên quan đến các outfit không hợp lệ (outfit không có tên — đã xác định ở bước làm sạch outfits). Điều này đảm bảo tính nhất quán: mỗi ảnh trong dataset đều phải thuộc về một outfit hợp lệ có đầy đủ thông tin.

---

## Cells liên quan: Cell 25 - 33

---

## Bước 1: Kiểm tra dữ liệu pictures

```python
pictures.info()
pictures.head()
```
Kiểm tra cấu trúc cột, số dòng, và kiểu dữ liệu.

---

## Bước 2: Xem lại danh sách null_outfit_ids

```python
null_outfit_ids
```
Biến `null_outfit_ids` được tạo ở bước làm sạch `outfits`, chứa các `id` của những outfit không có tên (null name). Đây là danh sách cần lọc ra khỏi `pictures`.

---

## Bước 3: Kiểm tra (debug) các dòng bị ảnh hưởng

```python
pictures_filter = pictures[pictures['outfit.id'].isin(null_outfit_ids)]
pictures_filter
```
Lọc ra các dòng trong `pictures` có `outfit.id` thuộc danh sách `null_outfit_ids` để xem xét trước khi xóa. Đây là bước kiểm tra không thay đổi dữ liệu.

---

## Bước 4: Loại bỏ các dòng liên quan đến outfit không hợp lệ

```python
pictures = pictures[~pictures['outfit.id'].isin(null_outfit_ids)]
```
- Dùng toán tử `~` (negation) để giữ lại các dòng có `outfit.id` **không nằm trong** `null_outfit_ids`.
- Kết quả gán lại vào `pictures` (inplace logic).

---

## Bước 5: Kiểm tra kết quả

```python
pictures.info()
```
So sánh số dòng trước và sau để xác nhận việc loại bỏ đã thành công.

---

## Kết quả

Sau bước làm sạch, `pictures` DataFrame:
- Không còn bản ghi nào có `outfit.id` thuộc `null_outfit_ids`.
- Tất cả ảnh đều thuộc về outfit có đầy đủ thông tin (`name` không null).

### Cấu trúc cột pictures sau làm sạch

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `outfit.id` | string | ID outfit (hợp lệ, có trong `outfits`) |
| `picture.id` | string | ID ảnh, định dạng `<outfit_id>.<số_thứ_tự>` |
| `file_name` | string | Tên file ảnh |
| `displayOrder` | int | Thứ tự hiển thị ảnh trong outfit |

---

## Tóm tắt luồng xử lý

```
pictures (raw)         null_outfit_ids (từ clean_outfits)
       |                          |
       +------------ filter ------+
       |
       pictures[~pictures['outfit.id'].isin(null_outfit_ids)]
       |
pictures (cleaned) -- chỉ còn outfit hợp lệ
```

---

## Lưu ý quan trọng

- `pictures` sau khi làm sạch sẽ được dùng trong:
  1. Bước kiểm tra nhất quán giữa `transactions` và `pictures` (cell 53).
  2. Bước tạo `grouped_sorted_df` để lấy danh sách ảnh theo từng outfit (cell 138).
  3. Tạo `image_list.txt` (cell 165).
- Cột `picture.id` có cấu trúc `"<outfit_id>.<seq>"`. Khi tạo đường dẫn embedding, chỉ lấy phần sau dấu chấm: `pic_id.split('.')[1]`.
