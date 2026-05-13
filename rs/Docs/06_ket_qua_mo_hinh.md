# 06 – Kết Quả Thực Nghiệm

## Metric đánh giá

- **k = 1, 3, 5, 10** (Top-K recommendation)
- **Recall@K**, **NDCG@K**

---

## Kết quả so sánh các chiến lược fusion

### Nhận xét chính

**Late Fusion** và **Weight Attention Fusion** đều cho kết quả **tốt hơn Early Fusion**, đặc biệt rõ ở **k=1**.

| Chiến lược | k=1 | Nhận xét |
|---|---|---|
| Early Fusion | Thấp nhất | Kết hợp quá sớm → mất thông tin đặc trưng riêng của từng modality |
| **Late Fusion** | **Tốt** | Tính similarity riêng cho image và text, gộp ở bước cuối |
| **Weight Attention Fusion** | **Tốt nhất** | Học trọng số tự động → tập trung vào modality quan trọng hơn |

---

## Lý giải

**Late Fusion tốt hơn Early Fusion** vì:
- Image embedding (MobileNetV2, 1280-D) và Text embedding (BERT, 768-D) có **không gian biểu diễn khác nhau**
- Gộp sớm (concatenate hoặc average) làm "pha loãng" đặc trưng của từng modality
- Late Fusion giữ nguyên từng similarity matrix rồi mới kết hợp → tín hiệu sạch hơn

**Weight Attention tốt hơn Late Fusion** vì:
- Trọng số được **học tự động** từ data thay vì đặt cứng α=0.5
- Model tự biết khi nào nên tin vào ảnh hơn, khi nào nên tin vào text hơn

---

## Khuyến nghị

Với dataset VCR 10k:
- Dùng **Late Fusion** làm baseline (đơn giản, ổn định)
- Dùng **Weight Attention** để đạt kết quả tốt nhất (phức tạp hơn một chút)
- Tránh dùng Early Fusion vì kết quả kém hơn đáng kể ở k=1
