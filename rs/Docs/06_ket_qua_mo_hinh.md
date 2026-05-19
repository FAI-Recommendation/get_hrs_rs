# 06 – Kết Quả Thực Nghiệm

## Metric đánh giá

- **k = 1, 5, 10, 20** (Top-K recommendation)
- **Recall@K**, **NDCG@K**

---

## Phạm vi thực nghiệm

Bộ thực nghiệm chính gồm **24 runs**: 3 models × 2 embeddings × 4 sim_type.

Trong đó, với nhóm multimodal, chỉ **2 fusion method** được đưa vào thực nghiệm chính:

| sim_type | Fusion method | Lý do chọn |
|---|---|---|
| `multimodal` | **Late Fusion** | Baseline ổn định, không cần tham số học thêm |
| `multimodal_attention` | **Weight Attention Fusion** | Học trọng số tự động, kết quả tốt nhất |

Hai method còn lại (`aggregation`, `pca`) đã thử sơ bộ nhưng không vượt trội → không đưa vào 24 thực nghiệm chính.

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
- Dùng **Late Fusion** (`sim_type=multimodal`) làm baseline (đơn giản, ổn định)
- Dùng **Weight Attention** (`sim_type=multimodal_attention`) để đạt kết quả tốt nhất (phức tạp hơn một chút)
- Tránh dùng Early Fusion vì kết quả kém hơn đáng kể ở k=1

---

## Mapping sim_type → Fusion method → Config

```bash
# Late Fusion (default)
python train.py --sim_type multimodal ...
# tương đương: MULTIMODAL_METHOD=late_fusion trong .env

# Weight Attention Fusion
python train.py --sim_type multimodal_attention ...
# tương đương: MULTIMODAL_METHOD=attention trong .env
```

Tham khảo chi tiết kiến trúc và cách chạy thực nghiệm tại:
- [`models_pyg_training_latest/00_README.md`](models_pyg_training_latest/00_README.md) — tổng quan 24 thực nghiệm
- [`models_pyg_training_latest/06_scripts_experiments.md`](models_pyg_training_latest/06_scripts_experiments.md) — bảng tracking
