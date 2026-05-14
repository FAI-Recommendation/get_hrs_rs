# Evaluation — 6 Metrics

## 1. Tong quan

Evaluation dung **foldout** strategy: voi moi user, xep hang tat ca items theo predicted score, roi so sanh voi ground truth.

### Data flow

```
model.predict(users)
    → scores (batch, n_items)
        → mask training items = -inf
            → argmax_top_k(scores, K)
                → compute 6 metrics
                    → mean over all users
```

## 2. Cac Metrics

### 2.1 Precision@K

```
Precision@K = |{relevant items in top-K}| / K
```

Trong so cac items duoc goi y (top-K), bao nhieu phan tram la dung?

### 2.2 Recall@K

```
Recall@K = |{relevant items in top-K}| / |{all relevant items}|
```

Trong so tat ca items ma user thuc su thich, bao nhieu phan tram duoc goi y trong top-K?

### 2.3 NDCG@K (Normalized Discounted Cumulative Gain)

```
DCG@K  = sum(1/log2(rank+1) for relevant items in top-K)
IDCG@K = sum(1/log2(i+1) for i in 1..min(K, |relevant|))
NDCG@K = DCG@K / IDCG@K
```

Danh gia chat luong thu tu: item dung o vi tri cao hon duoc thuong nhieu hon.

### 2.4 MAP@K (Mean Average Precision)

```
AP@K = (1/|relevant|) * sum(Precision@i * rel(i) for i in 1..K)
MAP  = mean(AP over all users)
```

Trung binh Precision tai moi vi tri co item dung. Phat model nang hon khi item dung o vi tri thap.

### 2.5 MRR@K (Mean Reciprocal Rank)

```
RR = 1 / rank_of_first_relevant_item
MRR = mean(RR over all users)
```

Vi tri cua item dung **dau tien** trong danh sach goi y. Huu ich khi chi can 1 goi y tot.

### 2.6 HR@K (Hit Ratio)

```
HR@K = 1 neu co it nhat 1 relevant item trong top-K, 0 neu khong
```

Don gian nhat: co hit hay khong?

## 3. Cach tinh chi tiet

### 3.1 File evaluate_foldout.py

```python
def eval_score_matrix_foldout(score_matrix, test_items, top_k=50):
    # Voi moi user:
    #   1. Lay top-K items theo score (dung heap)
    #   2. Tinh 6 metrics
    #   3. Tra ve array (n_users, top_k * 6)
```

Moi metric duoc tinh **tich luy** (cumulative) tai moi vi tri tu 1 den top_k:
- `precision = [P@1, P@2, ..., P@K]`
- `recall    = [R@1, R@2, ..., R@K]`
- ... tuong tu cho 4 metrics con lai

### 3.2 File batch_test.py

```python
def test(model, interaction_adj, similarity_adj, data, Ks, device):
    # 1. Batch users (eval_batch_size = 1024)
    # 2. model.predict() → scores (batch, n_items)
    # 3. Mask training items → -inf
    # 4. eval_score_matrix_foldout(scores, test_items, max_top)
    # 5. Mean over all users
    # 6. Reshape → [6, len(Ks)]
    # 7. Return dict: {precision, recall, ndcg, map, mrr, hit_ratio}
```

### 3.3 Masking Training Items

```python
for idx, user in enumerate(user_batch):
    train_items_off = data.train_items[user]
    rate_batch[idx][train_items_off] = -inf
```

Items da tuong tac trong training set bi dat score = -inf, de chung khong xuat hien trong top-K. Chi evaluate tren items **chua thay** trong training.

## 4. Cac K values

| K | Y nghia |
|---|---|
| 1 | Chi goi y 1 item — do chinh xac cao nhat |
| 5 | Top-5 — pho bien cho mobile app |
| 10 | Top-10 — pho bien cho web |
| 20 | Top-20 — can bang giua precision va recall |
| 50 | Top-50 — danh gia recall rong |

## 5. Output format

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

## 6. train_set_flag

| Flag | Evaluate tren | Muc dich |
|---|---|---|
| 0 | Test set | Do luong hieu qua thuc su (generalization) |
| 1 | Train set | Kiem tra model co hoc duoc khong (fitting) |

Trong training loop, ca 2 deu duoc chay moi eval_interval epochs.
