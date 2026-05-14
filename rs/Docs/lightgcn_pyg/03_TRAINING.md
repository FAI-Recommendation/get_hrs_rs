# Training Pipeline

## 1. Tong quan

```
parse_args() → Data() → get_norm_adj_mat() → CombiGCN() → Training Loop → Evaluation → Save
```

## 2. Loss Function — BPR (Bayesian Personalized Ranking)

### 2.1 MF Loss

```python
pos_scores = sum(user_emb * pos_item_emb, dim=1)   # dot product
neg_scores = sum(user_emb * neg_item_emb, dim=1)

mf_loss = mean(softplus(-(pos_scores - neg_scores)))
```

- `softplus(x) = log(1 + exp(x))` — smooth approximation cua ReLU
- Y nghia: day pos_scores len cao hon neg_scores
- Dung `softplus` thay vi `log-sigmoid` de on dinh hon (khong bi log(0))

### 2.2 Regularization Loss

```python
reg_loss = decay * (||u_pre||^2 + ||pos_pre||^2 + ||neg_pre||^2) / batch_size
```

- L2 regularization tren **pre-propagation embeddings** (truoc khi qua GCN)
- `decay` mac dinh: 1e-5
- Ngan overfitting, giu embeddings khong qua lon

### 2.3 Total Loss

```python
loss = mf_loss + reg_loss
```

## 3. Optimizer

```python
optimizer = Adam(model.parameters(), lr=0.001)
```

- Adam voi learning rate 0.001 (giong ban goc)
- Khong dung weight decay trong optimizer (da co L2 reg trong loss)

## 4. Training Loop

```
For each epoch:
    1. n_batch = n_train // batch_size + 1
    2. For each batch:
        a. Sample (users, pos_items, neg_items)      ← BPR sampling
        b. loss = model.forward(adj, sim_adj, u, p, n)
        c. loss.backward()
        d. optimizer.step()
    3. Log loss to TensorBoard
    4. If epoch % eval_interval == 0:
        a. Evaluate on train set
        b. Evaluate on test set
        c. Check early stopping
        d. Save best model
```

### 4.1 BPR Sampling

Moi batch:
- Chon `batch_size` users ngau nhien tu `exist_users`
- Voi moi user:
  - **Positive item**: random 1 item tu `train_items[user]`
  - **Negative item**: random 1 item KHONG co trong `train_items[user]`

```python
users, pos_items, neg_items = data.sample()
# users:     [u1, u2, u3, ...]       (batch_size,)
# pos_items: [p1, p2, p3, ...]       (batch_size,)
# neg_items: [n1, n2, n3, ...]       (batch_size,)
```

### 4.2 Batch Size

- Training: `--batch_size 1024` (mac dinh)
- Ban goc TF1 dung 32768, nhung voi dataset nho (VCR 10k) thi 1024 hop ly hon
- Co the tang len 2048 hoac 4096 neu GPU du VRAM

## 5. Early Stopping

```python
cur_best, stopping_step, should_stop = early_stopping(
    log_value=ret['recall'],        # metric hien tai
    best_value=cur_best,            # metric tot nhat
    stopping_step=stopping_step,    # dem so buoc khong cai thien
    expected_order='acc',           # 'acc' = lon hon la tot hon
    flag_step=5,                    # dung sau 5 eval intervals khong cai thien
)
```

- Theo doi `recall` tai tat ca K values
- Neu **tat ca** K deu khong cai thien → tang `stopping_step`
- Neu bat ky K nao cai thien → reset `stopping_step = 0`
- Dung khi `stopping_step >= flag_step` (mac dinh 5)
- Voi `eval_interval=10` va `flag_step=5` → dung sau **50 epochs** khong cai thien

## 6. Model Saving

Khi recall[0] (recall@K_min) dat best:

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    'args': vars(args),
}, 'weights/<sim_type>/layers_3_dim_64/lr_0.001_reg_1e-05/best_model.pt')
```

### Load lai:

```python
checkpoint = torch.load('weights/img_only/.../best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 7. TensorBoard

```bash
tensorboard --logdir rs/lightgcn_pyg/tensorboard/
```

Metrics duoc log:
- `train/loss`, `train/mf_loss`, `train/reg_loss` — moi epoch
- `test/recall@K`, `test/precision@K`, `test/ndcg@K`, etc. — moi eval_interval

## 8. Hyperparameters mac dinh

| Tham so | Gia tri | Mo ta |
|---|---|---|
| `embed_size` | 64 | Chieu embedding |
| `layer_size` | [64,64,64] | 3 layers |
| `lr` | 0.001 | Learning rate |
| `regs` | [1e-5] | L2 decay |
| `batch_size` | 1024 | Training batch |
| `epoch` | 1000 | Max epochs |
| `eval_interval` | 10 | Evaluate moi 10 epoch |
| `early_stop_steps` | 5 | Patience |
| `node_dropout` | 0.0 | Off by default |
| `Ks` | [1,5,10,20,50] | Top-K evaluation |

## 9. So sanh TF1 vs PyG training

| | TF1 (ban goc) | PyG (ban moi) |
|---|---|---|
| Session | `tf.Session` | Khong can |
| Sampling | Thread + `tf.device(cpu)` | Python threading truc tiep |
| Forward | `sess.run(feed_dict=...)` | `model(adj, sim, u, p, n)` |
| Backward | `tf.train.AdamOptimizer.minimize` | `loss.backward() + optimizer.step()` |
| GPU | `tf.ConfigProto(allow_growth)` | `model.to(device)` |
| TensorBoard | `tf.summary.FileWriter` | `torch.utils.tensorboard.SummaryWriter` |
| Save | `tf.train.Saver` | `torch.save(state_dict)` |
