# Glossary — Evaluation trong Recommender System

## Hyperparameters (config.json)

| Từ khóa | Giải thích |
|---|---|
| `lr` | **Learning rate** — tốc độ cập nhật trọng số. Ví dụ: `0.001` |
| `decay` | **Weight decay / L2 regularization** — hệ số phạt để tránh overfitting. Ví dụ: `1e-4` |
| `n_users` / `n_items` | Số user và item trong tập dữ liệu |
| `n_layers` | Số lớp graph convolution trong GCN |
| `sim_type` | Loại similarity dùng để xây đồ thị. Các giá trị: `img_only`, `tfidf`, `multimodal`, `multimodal_attention` |
| `batch_size` | Số mẫu mỗi lần cập nhật gradient |
| `embed_size` | Chiều của embedding vector |

---

## Loss (train/)

| Từ khóa | Giải thích |
|---|---|
| `train/loss` | Tổng loss = `mf_loss + reg_loss` |
| `train/mf_loss` | **Matrix Factorization loss** — BPR loss đo độ phù hợp ranking giữa positive và negative sample |
| `train/reg_loss` | **Regularization loss** — phần phạt L2 trên embedding, tránh overfitting |

---

## Evaluation Metrics (test/) — đo tại K = 1, 5, 10, 20

> Model gợi ý top-K items cho mỗi user. Các metrics đo chất lượng danh sách đó.

| Từ khóa | Giải thích |
|---|---|
| `recall@K` | Tỷ lệ item đúng được tìm thấy trong top-K / tổng item đúng của user. Đo **độ bao phủ** |
| `precision@K` | Tỷ lệ item đúng trong top-K / K. Đo **độ chính xác** của danh sách gợi ý |
| `hit_ratio@K` | = 1 nếu có ít nhất 1 item đúng trong top-K, ngược lại = 0. Đo **xác suất hit** trung bình |
| `ndcg@K` | **Normalized Discounted Cumulative Gain** — đo chất lượng ranking, item đúng ở vị trí cao được thưởng nhiều hơn |
| `map@K` | **Mean Average Precision** — trung bình precision tại mỗi vị trí có item đúng trong top-K |
| `mrr@K` | **Mean Reciprocal Rank** — trung bình `1/rank` của item đúng đầu tiên trong top-K. Nhấn mạnh item đúng đứng càng đầu càng tốt |

---

## Metadata (summary.json)

| Từ khóa | Giải thích |
|---|---|
| `best_epoch` | Epoch mà model đạt NDCG tốt nhất trên val set |
| `best_ndcg@K0` | Giá trị NDCG tốt nhất (trên val) dùng để chọn checkpoint |
| `best_recall@K0` | Recall tương ứng tại `best_epoch` |
| `epoch` | Tổng số epoch đã train |
| `_step` | Số bước wandb log |
| `_runtime` | Thời gian chạy tính bằng giây |
| `state` | Trạng thái run: `finished` = hoàn thành bình thường |
| `run_name` | Tên run encode toàn bộ config để dễ phân biệt các thực nghiệm |

---

## Thứ tự ưu tiên khi so sánh models

`NDCG@10` > `Recall@10` > `Precision@10`

NDCG là metric chính vì đo cả chất lượng ranking (vị trí item đúng trong danh sách), không chỉ đơn thuần đếm số lượng item đúng.
