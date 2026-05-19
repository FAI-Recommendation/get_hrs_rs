# 04 — Training Pipeline

---

## Tổng quan

```
parse_args() → Data() → get_norm_adj_mat() → Model() → Training Loop → Evaluation → Save
```

---

## Loss Function — BPR (Bayesian Personalized Ranking)

### MF Loss

```python
pos_scores = sum(user_emb * pos_item_emb, dim=1)   # dot product
neg_scores = sum(user_emb * neg_item_emb, dim=1)

mf_loss = mean(softplus(-(pos_scores - neg_scores)))
```

- `softplus(x) = log(1 + exp(x))` — smooth approximation của ReLU
- Ý nghĩa: đẩy pos_scores lên cao hơn neg_scores
- Dùng `softplus` thay vì `log-sigmoid` để ổn định hơn (không bị log(0))

### Regularization Loss

```python
reg_loss = decay * (||u_pre||^2 + ||pos_pre||^2 + ||neg_pre||^2) / batch_size
```

L2 regularization trên **pre-propagation embeddings** (trước khi qua GCN). Ngăn overfitting.

### BM3 / FREEDOM thêm Contrastive Loss

| Model | Thêm loss |
|---|---|
| BM3 | `+ cl_weight × bootstrap_cl_loss` |
| FREEDOM | `+ cl_weight × InfoNCE` |

### Total Loss

```python
loss = mf_loss + reg_loss  # CombiGCN
loss = mf_loss + reg_loss + cl_weight * cl_loss  # BM3, FREEDOM
```

---

## Optimizer

```python
optimizer = Adam(model.parameters(), lr=0.001)
```

Adam với learning rate 0.001. Không dùng weight decay trong optimizer (đã có L2 reg trong loss).

---

## Training Loop

```
For each epoch:
    1. n_batch = n_train // batch_size + 1
    2. For each batch:
        a. Sample (users, pos_items, neg_items)      ← BPR sampling
        b. loss = model.forward(adj, sim_adj, u, p, n)
        c. loss.backward()
        d. optimizer.step()
    3. Log loss to TensorBoard / wandb
    4. If epoch % eval_interval == 0:
        a. Evaluate on train set
        b. Evaluate on test set
        c. Check early stopping
        d. Save best model
```

### BPR Sampling

```python
users, pos_items, neg_items = data.sample()
# users:     [u1, u2, u3, ...]       (batch_size,)
# pos_items: [p1, p2, p3, ...]       (batch_size,)
# neg_items: [n1, n2, n3, ...]       (batch_size,)
```

---

## Early Stopping

```python
cur_best, stopping_step, should_stop = early_stopping(
    log_value=ret['recall'],        # metric hiện tại
    best_value=cur_best,            # metric tốt nhất
    stopping_step=stopping_step,    # đếm số bước không cải thiện
    expected_order='acc',           # 'acc' = lớn hơn là tốt hơn
    flag_step=5,                    # dừng sau 5 eval intervals không cải thiện
)
```

- Nếu **tất cả** K đều không cải thiện → tăng `stopping_step`
- Nếu bất kỳ K nào cải thiện → reset `stopping_step = 0`
- Với `eval_interval=10` và `flag_step=5` → dừng sau **50 epochs** không cải thiện

---

## Model Saving

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    'args': vars(args),
}, 'weights/<sim_type>/layers_3_dim_64/lr_0.001_reg_1e-05/best_model.pt')
```

Load lại:

```python
checkpoint = torch.load('weights/img_only/.../best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## TensorBoard

```bash
tensorboard --logdir rs/lightgcn_pyg/tensorboard/
```

Metrics được log:
- `train/loss`, `train/mf_loss`, `train/reg_loss` — mỗi epoch
- `test/recall@K`, `test/precision@K`, `test/ndcg@K`, ... — mỗi eval_interval

---

## Hyperparameters mặc định

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `embed_size` | 64 (test) / 512 (full) | Chiều embedding |
| `layer_size` | [64,64,64] / [512,512,512,512] | Số layers |
| `lr` | 0.001 | Learning rate |
| `regs` | [1e-5] / [1e-4] | L2 decay |
| `batch_size` | 1024 (test) / 8192 (full) | Training batch |
| `epoch` | 20 (test) / 1000 (full) | Max epochs |
| `eval_interval` | 10 (test) / 40 (full) | Evaluate mỗi N epoch |
| `early_stop_steps` | 3 (test) / 5 (full) | Patience |
| `node_dropout` | 0.0 | Off by default |
| `Ks` | [1,5,10,20] | Top-K evaluation |

---

## So sánh TF1 vs PyG

| | TF1 (bản gốc) | PyG (bản mới) |
|---|---|---|
| Session | `tf.Session` | Không cần |
| Forward | `sess.run(feed_dict=...)` | `model(adj, sim, u, p, n)` |
| Backward | `tf.train.AdamOptimizer.minimize` | `loss.backward() + optimizer.step()` |
| GPU | `tf.ConfigProto(allow_growth)` | `model.to(device)` |
| TensorBoard | `tf.summary.FileWriter` | `torch.utils.tensorboard.SummaryWriter` |
| Save | `tf.train.Saver` | `torch.save(state_dict)` |

---

## Output sau khi training

```
rs/lightgcn_pyg/
├── weights/
│   ├── none/layers_3_dim_64/lr_0.001_reg_1e-05/best_model.pt
│   ├── img_only/.../best_model.pt
│   ├── multimodal/.../best_model.pt
│   └── tfidf/.../best_model.pt
│
├── output/
│   ├── none/combigcn.result
│   ├── img_only/combigcn.result
│   ├── multimodal/combigcn.result
│   └── tfidf/combigcn.result
│
└── tensorboard/
    ├── none_layers3_dim64_lr0.001_reg1e-05/
    └── ...
```
