# 00 — Tổng Quan Pipeline: Raw Data → Train/Test

Tài liệu này mô tả toàn bộ hành trình dữ liệu từ 3 file CSV gốc (VCR dataset) đến 2 file `train.txt` / `test.txt` sẵn sàng cho model. Mỗi bước đều có dữ liệu bị loại bỏ — tài liệu giải thích **bị mất ở đâu, bị mất bao nhiêu, và tại sao**.

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
| Bước 1 — lọc ảnh | Outfit chỉ có ảnh displayOrder ≠ 0 → không lấy được | Outfit chỉ có ảnh chụp lưng |
| Bước 1 — merge | Outfit không có name hợp lệ trong outfits.csv → loại | Outfit null name |
| Bước 1 — sample | User có < 3 GD bị loại khỏi sampling pool | User mới chỉ thuê 1 lần |
| Bước 2 — N-Core | User có < 5 GD trong 10k sample bị loại | User chỉ có 3-4 GD sau sample |
| Bước 2 — N-Core (gián tiếp) | Item chỉ được thuê bởi user bị loại → item cũng biến mất | 2.223 → 2.194 items (-29) |

Xem chi tiết từng bước tại: [01_chi_tiet_tung_buoc.md](01_chi_tiet_tung_buoc.md)

---

## Vì sao cả train lẫn test đều có đúng 553 users?

Đây là điểm **dễ hiểu nhầm nhất**. Không phải 80% users vào train và 20% users vào test.

```
Cách hiểu SAI:
  553 users → 442 vào train, 111 vào test

Cách code thực sự làm:
  Mỗi user được chia độc lập theo thời gian:
    User A (10 GD): GD 1-8 → train   |  GD 9-10 → test
    User B (5 GD):  GD 1-4 → train   |  GD 5    → test
    User C (20 GD): GD 1-16 → train  |  GD 17-20 → test
```

Kết quả: **mỗi user đều xuất hiện trong cả train VÀ test**.

Lý do phải làm vậy:
- Nếu user không có trong **train** → model không học được preference của họ → không thể gợi ý cho họ.
- Nếu user không có trong **test** → không evaluate được model trên họ.

Đây là chuẩn **per-user temporal split** trong Recommender System, tránh được data leakage và đảm bảo đánh giá công bằng.
