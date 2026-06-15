# data_evaluate — Evaluation & Visualization

Tổng hợp kết quả train **24 cấu hình** (3 models × 2 encoders × 4 sim_types),
tính 6 metrics tại K=1/5/10/20, và dựng hệ thống **82 biểu đồ** so sánh để trả
lời các câu hỏi nghiên cứu (encoder nào tốt, sim_type nào tốt, model nào tốt).

---

## Mô tả các file

| File / thư mục | Vai trò |
|---|---|
| `evaluate.ipynb` | Notebook chính: load logs → tính metrics → dựng toàn bộ biểu đồ |
| `plot_tier1.py` | Vẽ biểu đồ Tier 1: best-config per model, so sánh tổng thể |
| `plot_tier2.py` | Vẽ biểu đồ Tier 2: ablation per model (sim_type × encoder) |
| `load_data.py` | Load & parse logs từ `data_wandb/` thành DataFrame |
| `reference_code.py` | Code tham khảo (helper plotting) |
| `EVALUATION_GUIDE.md` | Hướng dẫn chi tiết quy trình đánh giá |
| `data_wandb/` | Raw logs từ WandB của 24 runs + `all_runs_summary.csv` |
| `charts/` | 82 biểu đồ PNG output (`Figure_01` … `Figure_82`) |
| `docs/` | Tài liệu bổ sung |
| `backup/`, `raw_gpuDocker/` | Backup logs & cấu hình Docker GPU |

---

## 6 Metrics × 4 K

| Metric | Ý nghĩa |
|---|---|
| `recall@K` | % ground truth tìm được trong top-K |
| `precision@K` | % items trong top-K là đúng |
| `ndcg@K` | Chất lượng ranking (item đúng ở vị trí cao được thưởng) |
| `map@K` | Mean Average Precision |
| `mrr@K` | Mean Reciprocal Rank (vị trí hit đầu tiên) |
| `hit_ratio@K` | Có ít nhất 1 item đúng trong top-K không |

K values: **1, 5, 10, 20**. Báo cáo phân tích chính ở **K=5** (dataset nhỏ, ~3.8
test items/user → K=5 phân biệt model rõ, ít bão hòa), giữ **K=10** để so với paper.

---

## Cách dùng

```bash
# Cách 1: chạy notebook (sinh toàn bộ 82 biểu đồ)
jupyter notebook evaluate.ipynb

# Cách 2: chạy script vẽ riêng từng tier
python plot_tier1.py    # best-config comparison
python plot_tier2.py    # per-model ablation
```

Input đọc từ `data_wandb/` (logs 24 runs), output ghi ra `charts/`.

---

## Hệ thống biểu đồ (`charts/`)

82 hình được đánh số `Figure_01` → `Figure_82`, nhóm theo metric và mục đích:

| Nhóm | Nội dung |
|---|---|
| Context (04, 07, …) | Heatmap/lineplot tổng quan 24 configs |
| Per-encoder (46, 47, 74, 75) | So sánh CLIP vs MobileNetV2 |
| Ablation (48–71) | Per-model × 4 sim_type × 2 encoder |
| Best-vs-Best (72–81) | So sánh 3 best config |
| Summary (82) | Best config theo từng metric |

---

## Kết quả chính

| Câu hỏi | Kết luận |
|---|---|
| Encoder nào tốt? | **MobileNetV2** — tạo best config cho cả 3 model |
| Sim_type nào tốt? | **`multimodal` (late fusion)**; attention chỉ giúp model yếu |
| Model nào tốt? | **BM3** — vượt CombiGCN +18% @K=5, vượt FREEDOM >2× |
| Thứ hạng | `BM3 > CombiGCN > FREEDOM` (ổn định qua mọi K, metric) |

**Cấu hình khuyến nghị:** `BM3 · MobileNetV2 · multimodal` — NDCG@5 = 0.0162, NDCG@10 = 0.0186.

> Phân tích đầy đủ: [`../report/BAO_CAO_TIEN_DO_EVALUATION.md`](../report/BAO_CAO_TIEN_DO_EVALUATION.md)

---

## Liên quan

- Dữ liệu đánh giá: [`../rs/get10k_data/`](../rs/get10k_data/)
- Code train: [`../rs/lightgcn_pyg/`](../rs/lightgcn_pyg/)
