# 05 — Data Loading, Evaluation & Scripts

## Data class (`utility/load_data.py`)

### Init — doc du lieu

```python
Data(path, batch_size)
```

Doc 3 file:
- `train.txt` — format: `uid item1 item2 ...` (moi dong 1 user)
- `test.txt` — format tuong tu
- `items_features.csv` — cac cot: feature1 (text), feature2 (text), feature3 (image embedding string)

Xay dung:
- `R` — sparse matrix [n_users, n_items] (interaction)
- `U = R @ R.T` — user co-occurrence
- `I = R.T @ R` — item co-occurrence
- `train_items`, `test_set` — dict uid → [item_ids]

### BPR Sampling

```python
users, pos_items, neg_items = data.sample()
```

Moi user:
- Random 1 positive item (tu train set)
- Random 1 negative item (khong trong train set)
- Batch = 8192 users/step

### Raw Embeddings (BM3 / FREEDOM)

```python
image_feats, text_feats = data.get_raw_embeddings()
```

Doc tu:
- `{data_path}/image_embeddings.npy` — shape [n_items, img_dim]
- `{data_path}/text_embeddings.npy` — shape [n_items, txt_dim]

> CombiGCN KHONG dung raw embeddings — no dung similarity matrices.

### Adjacency matrices

```python
matrices = data.get_norm_adj_mat(sim_type, multimodal_method)
```

Tra ve 8 matrices (chi build cai can thiet, con lai = None):

| Index | Ten | Shape | Dung cho |
|---|---|---|---|
| 0 | interaction | [N, N] (N=users+items) | Tat ca models |
| 1 | social | [users, users] | Khong dung |
| 2 | similar_users | [users, users] | Khong dung |
| 3 | tfidf_item | [items, items] | CombiGCN sim_type=tfidf |
| 4 | bert_item | [items, items] | Khong dung |
| 5 | full_bert_item | [items, items] | Khong dung |
| 6 | multimodal | [items, items] | CombiGCN sim_type=multimodal |
| 7 | img_only | [items, items] | CombiGCN sim_type=img_only |

**Caching:** Moi matrix duoc save/load tu file `.npz` trong data_path. Lan dau build, lan sau load.

**Normalization:** Tat ca deu dung symmetric normalization `D^-0.5 A D^-0.5`.

### GPU Cosine Similarity

```python
cosine_sim_gpu(a, b=None)  # Nhanh hon sklearn 20-50x
```

- Input: numpy array hoac scipy sparse
- Batch 512 tren GPU de tranh OOM
- Dung cho tat ca similarity matrices

---

## Evaluation (`utility/batch_test.py`)

### Flow

```python
ret = test(model, interaction_adj, similarity_adj, data, Ks, device)
```

1. Lay danh sach test users
2. Chia batch (1024 users/batch)
3. `model.predict(adj, sim_adj, users)` → scores [batch, n_items]
4. Mask training items = -inf (khong recommend item da mua)
5. `eval_score_matrix_foldout(scores, test_items, max_K)`
6. Mean tren tat ca users → final metrics

### 6 Metrics (`evaluator/evaluate_foldout.py`)

| Metric | Y nghia | Cong thuc |
|---|---|---|
| **Precision@K** | Ty le item dung trong top-K | hits / K |
| **Recall@K** | Ty le ground truth tim duoc | hits / \|GT\| |
| **NDCG@K** | Chat luong ranking (uu tien top) | DCG / IDCG |
| **MAP@K** | Mean average precision | sum(P@k * rel_k) / \|GT\| |
| **MRR@K** | Vi tri item dung dau tien | 1 / rank_of_first_hit |
| **Hit Ratio@K** | Co it nhat 1 item dung? | 1 neu co hit, 0 neu khong |

> Tat ca metrics duoc tinh **cumulative** (K=1 den max_K), sau do lay tai cac moc K=[1,5,10,20].

---

## Scripts

### Cau truc thu muc scripts

```
scripts/
├── 01_run_all_clip/          # CombiGCN × CLIP × 4 sim_types
├── 01_test_clip/             # Test CombiGCN × CLIP
├── 02_run_all_mbnv2/         # CombiGCN × MBNv2 × 4 sim_types
├── 02_test_mbnv2/            # Test CombiGCN × MBNv2
├── 03_run_bm3_clip/          # BM3 × CLIP × 4 sim_types
├── 03_test_bm3_clip/
├── 04_run_bm3_mbnv2/         # BM3 × MBNv2 × 4 sim_types
├── 04_test_bm3_mbnv2/
├── 05_run_freedom_clip/      # FREEDOM × CLIP × 4 sim_types
├── 05_test_freedom_clip/
├── 06_run_freedom_mbnv2/     # FREEDOM × MBNv2 × 4 sim_types
└── 06_test_freedom_mbnv2/
```

### Moi folder chua:

```
run_all.sh                    # Chay 4 sim_types lien tiep
run_img_only.sh
run_tfidf.sh
run_multimodal.sh
run_multimodal_attention.sh   # (chi BM3/FREEDOM)
```

### Template 1 run script (vi du BM3 + CLIP + multimodal):

```bash
python3 train.py \
    --model bm3 \
    --embed_type clip \
    --sim_type multimodal \
    --data_path ../get10k_data/clip_10k_sample \
    --dataset "" \
    --embed_size 512 \
    --layer_size "[512,512,512,512]" \
    --lr 0.001 \
    --regs "[1e-4]" \
    --batch_size 8192 \
    --epoch 1000 \
    --eval_interval 40 \
    --early_stop_steps 0 \
    --Ks "[1,5,10,20]" \
    --bm3_momentum 0.995 \
    --bm3_cl_weight 0.2 \
    --save_flag 1 \
    --checkpoint_interval 200 \
    --gpu_id 0
```

### Tong so experiments: 24

```
3 models × 2 encoders × 4 sim_types = 24 runs
Moi run: 1000 epochs, eval moi 40 epochs → 25 diem danh gia
```

### Data paths theo encoder:

| Encoder | Data path |
|---|---|
| CLIP | `../get10k_data/clip_10k_sample` |
| MBNv2 | `../get10k_data/mbnv2_10k_sample` |

---

## Logging & Output

### TensorBoard (luon bat)

```
tensorboard/
└── {model}_{sim_type}_layers{L}_dim{D}_lr{lr}_reg{decay}/
    └── events.out.tfevents.*
```

### WandB (auto-detect tu env)

- Detect `WANDB_API_KEY` tu .env
- Project: `combigcn-rs` (hoac tu `WANDB_PROJECT`)
- Log: train/loss, test/recall@K, test/ndcg@K, ...

### Output files

```
output/{model_tag}/
└── {model}.result           # Append-mode text file

weights/{model_tag}/layers_{L}_dim_{D}/lr_{lr}_reg_{decay}/
├── best_model.pt            # state_dict + metrics + args
├── best_metrics.json        # JSON summary
└── checkpoint_epoch{N}.pt   # Periodic checkpoints (moi 200 epochs)
```

### HuggingFace Hub (optional)

- Detect `HF_TOKEN` + `HF_REPO_ID` tu .env
- Push toan bo weights folder sau khi training xong
- 1 lan duy nhat cuoi cung (khong push moi epoch)
