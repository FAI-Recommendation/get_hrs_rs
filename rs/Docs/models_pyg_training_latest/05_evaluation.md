# 05 — Evaluation: 6 Metrics & Glossary

---

## Tổng quan

Evaluation dùng **foldout** strategy: với mỗi user, xếp hạng tất cả items theo predicted score, rồi so sánh với ground truth.

```
model.predict(users)
    → scores (batch, n_items)
        → mask training items = -inf
            → argmax_top_k(scores, K)
                → compute 6 metrics
                    → mean over all users
```

---

## 6 Metrics

### Precision@K

```
Precision@K = |{relevant items in top-K}| / K
```

Trong số các items được gợi ý (top-K), bao nhiêu phần trăm là đúng?

### Recall@K

```
Recall@K = |{relevant items in top-K}| / |{all relevant items}|
```

Trong số tất cả items mà user thực sự thích, bao nhiêu phần trăm được gợi ý trong top-K?

### NDCG@K (Normalized Discounted Cumulative Gain)

```
DCG@K  = sum(1/log2(rank+1) for relevant items in top-K)
IDCG@K = sum(1/log2(i+1) for i in 1..min(K, |relevant|))
NDCG@K = DCG@K / IDCG@K
```

Đánh giá chất lượng thứ tự: item đúng ở vị trí cao hơn được thưởng nhiều hơn. **Metric chính để so sánh.**

### MAP@K (Mean Average Precision)

```
AP@K = (1/|relevant|) * sum(Precision@i * rel(i) for i in 1..K)
MAP  = mean(AP over all users)
```

Trung bình Precision tại mỗi vị trí có item đúng. Phạt model nặng hơn khi item đúng ở vị trí thấp.

### MRR@K (Mean Reciprocal Rank)

```
RR = 1 / rank_of_first_relevant_item
MRR = mean(RR over all users)
```

Vị trí của item đúng **đầu tiên** trong danh sách gợi ý. Hữu ích khi chỉ cần 1 gợi ý tốt.

### HR@K (Hit Ratio)

```
HR@K = 1 nếu có ít nhất 1 relevant item trong top-K, 0 nếu không
```

Đơn giản nhất: có hit hay không?

---

## Cách tính chi tiết

### File evaluate_foldout.py

```python
def eval_score_matrix_foldout(score_matrix, test_items, top_k=50):
    # Với mỗi user:
    #   1. Lấy top-K items theo score (dùng heap)
    #   2. Tính 6 metrics (tích lũy từ rank 1 đến K)
    #   3. Trả về array (n_users, top_k * 6)
```

Mỗi metric được tính **tích lũy**:
- `precision = [P@1, P@2, ..., P@K]`
- `recall    = [R@1, R@2, ..., R@K]`
- ... tương tự cho 4 metrics còn lại

### File batch_test.py

```python
def test(model, interaction_adj, similarity_adj, data, Ks, device):
    # 1. Batch users (eval_batch_size = 1024)
    # 2. model.predict() → scores (batch, n_items)
    # 3. Mask training items → -inf
    # 4. eval_score_matrix_foldout(scores, test_items, max_top)
    # 5. Mean over all users
    # 6. Return dict: {precision, recall, ndcg, map, mrr, hit_ratio}
```

### Masking Training Items

```python
for idx, user in enumerate(user_batch):
    train_items_off = data.train_items[user]
    rate_batch[idx][train_items_off] = -inf
```

Items đã tương tác trong training set bị đặt score = -inf → chỉ evaluate trên items **chưa thấy** trong training.

---

## Các K values

| K | Ý nghĩa |
|---|---|
| 1 | Chỉ gợi ý 1 item — độ chính xác cao nhất |
| 5 | Top-5 — phổ biến cho mobile app |
| 10 | Top-10 — phổ biến cho web |
| 20 | Top-20 — cân bằng giữa precision và recall |
| 50 | Top-50 — đánh giá recall rộng |

---

## Output format

```python
result = {
    'precision':  [P@1, P@5, P@10, P@20, P@50],
    'recall':     [R@1, R@5, R@10, R@20, R@50],
    'ndcg':       [N@1, N@5, N@10, N@20, N@50],
    'map':        [M@1, M@5, M@10, M@20, M@50],
    'mrr':        [MRR@1, MRR@5, MRR@10, MRR@20, MRR@50],
    'hit_ratio':  [HR@1, HR@5, HR@10, HR@20, HR@50],
}
```

---

## Thứ tự ưu tiên khi so sánh models

```
NDCG@10 > Recall@10 > Precision@10
```

NDCG là metric chính vì đo cả chất lượng ranking (vị trí item đúng trong danh sách), không chỉ đơn thuần đếm số lượng item đúng.

---

## Glossary

### Hyperparameters

| Từ khóa | Giải thích |
|---|---|
| `lr` | Learning rate — tốc độ cập nhật trọng số |
| `decay` | Weight decay / L2 regularization |
| `n_users` / `n_items` | Số user và item trong tập dữ liệu |
| `n_layers` | Số lớp graph convolution |
| `sim_type` | Loại similarity dùng để xây đồ thị |
| `batch_size` | Số mẫu mỗi lần cập nhật gradient |
| `embed_size` | Chiều của embedding vector |

### Loss (train/)

| Từ khóa | Giải thích |
|---|---|
| `train/loss` | Tổng loss = `mf_loss + reg_loss` (+ cl_loss nếu có) |
| `train/mf_loss` | BPR loss — đo độ phù hợp ranking giữa positive và negative sample |
| `train/reg_loss` | Regularization loss — phần phạt L2 trên embedding |

### Metadata (summary.json / wandb)

| Từ khóa | Giải thích |
|---|---|
| `best_epoch` | Epoch mà model đạt NDCG tốt nhất |
| `best_ndcg@K0` | Giá trị NDCG tốt nhất dùng để chọn checkpoint |
| `best_recall@K0` | Recall tương ứng tại `best_epoch` |
| `state` | Trạng thái run: `finished` = hoàn thành bình thường |
| `run_name` | Tên run encode toàn bộ config |

### train_set_flag

| Flag | Evaluate trên | Mục đích |
|---|---|---|
| 0 | Test set | Đo lường hiệu quả thực sự (generalization) |
| 1 | Train set | Kiểm tra model có học được không (fitting) |
