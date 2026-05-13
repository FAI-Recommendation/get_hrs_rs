# 02 – Thuật Toán Split 10k Sample (`slipt_10k_sample.py`)

## Mục tiêu

Từ toàn bộ VCR dataset (~64k giao dịch, ~50k ảnh), lấy ra một mẫu **~10k giao dịch** đại diện, đảm bảo:
- Mỗi outfit có đúng **1 ảnh chính** (`displayOrder == 0`)
- Số outfit == số ảnh (không thừa, không thiếu)
- Giữ nguyên 100% cột gốc của 3 file CSV

---

## Luồng 5 bước

```
[3 file CSV gốc]
       ↓
Bước 1. pick_main_picture_rows()
        lọc displayOrder == 0
        drop_duplicates(outfit.id)      ← 1 outfit = 1 ảnh duy nhất
        → pictures_main
       ↓
Bước 2. merge_sources()
        interactions INNER JOIN pictures_main ON outfit.id
                     INNER JOIN outfits        ON outfit.id = id
        → chỉ giữ outfit có đủ ảnh + text hợp lệ
       ↓
Bước 3. sample_by_users()
        lọc user có >= 3 giao dịch
        cộng dồn giao dịch từng user (shuffle ngẫu nhiên)
        cho đến khi đủ >= 10.000 rows
        → sampled (~10k rows)
       ↓
Bước 4. Truy ngược về interactions_orig
        dùng _orig_row_id → lấy đúng dòng gốc
        → giữ nguyên 100% cột gốc, xóa cột tạm
       ↓
Bước 5. build_outputs()
        sampled_outfit_ids = outfit.id có trong sample
        pictures_sampled = pictures_main WHERE outfit.id IN sampled_outfit_ids
        outfits_sampled  = outfits_orig  WHERE id         IN sampled_outfit_ids
        → outfit count == picture count == 2223
```

---

## Tại sao User-centric Sampling?

Lấy mẫu theo **user** thay vì random toàn bộ vì:

| Vấn đề nếu random | Giải pháp user-centric |
|---|---|
| User có thể chỉ còn 1 giao dịch → rơi vào test, không học được ở train | Lấy **toàn bộ** giao dịch của mỗi user được chọn |
| Data mất cân bằng theo user | Đảm bảo mỗi user có đủ lịch sử |

---

## Ràng buộc quan trọng

| Rule | Lý do |
|---|---|
| Tên file output = tên gốc + `_10k` | Không đổi convention |
| Không thêm/xóa cột | Notebook preprocessing đọc đúng schema |
| `outfit count == picture count` | 1 outfit = 1 ảnh = 1 embedding → ma trận không bị lỗi chiều |

---

## Output

| File | Rows | Ghi chú |
|---|---|---|
| `user_activity_triplets_10k.csv` | ~10.000 | Giữ nguyên 4 cột gốc |
| `picture_triplets_10k.csv` | 2.223 | Chỉ displayOrder==0, 1 row/outfit |
| `outfits_10k.csv` | 2.223 | Giữ nguyên 11 cột gốc |
