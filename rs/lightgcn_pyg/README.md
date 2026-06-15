# lightgcn_pyg — Models & Training

Code train/eval 3 model gợi ý thời trang đa phương thức, đều dùng **LightGCN**
làm backbone collaborative filtering. Viết bằng PyTorch + PyTorch Geometric
(`torch_sparse`).

---

## 3 Models

| Model | Ý tưởng cốt lõi | Loss | File |
|---|---|---|---|
| **CombiGCN** | Dual-graph: kết hợp user-item interaction graph + item-item similarity graph (precompute). Fusion mỗi layer. | BPR + L2 | [`models/combigcn.py`](models/combigcn.py) |
| **BM3** | Bootstrap contrastive learning (kiểu BYOL): online encoder + EMA target + predictor. Không cần negative samples. | BPR + Bootstrap CL | [`models/bm3.py`](models/bm3.py) |
| **FREEDOM** | Frozen item-item kNN graph (build 1 lần từ feature, đóng băng). Dual propagation CF + content. | BPR + InfoNCE | [`models/freedom.py`](models/freedom.py) |

> Docs workflow chi tiết từng model: `../Docs/model3_v2/` và `../Docs/prompt3models/`.

## 4 sim_types (cách kết hợp đặc trưng)

| sim_type | Mô tả |
|---|---|
| `img_only` | Chỉ dùng đặc trưng ảnh |
| `tfidf` | Chỉ dùng đặc trưng text (TF-IDF) |
| `multimodal` | Image + text, **late fusion** (trung bình) |
| `multimodal_attention` | Image + text, **weight attention fusion** (học weight) |

---

## Cấu trúc thư mục

```
lightgcn_pyg/
├── train.py              # Entry point: train + eval + log + checkpoint
├── model.py              # LightGCN backbone gốc
├── models/
│   ├── combigcn.py       # CombiGCN (dual-graph)
│   ├── bm3.py            # BM3 (bootstrap CL + EMA)
│   └── freedom.py        # FREEDOM (frozen kNN graph + InfoNCE)
├── utility/
│   ├── load_data.py      # Data class: load adjacency, raw embeddings, BPR sampling
│   ├── parser.py         # Argument parser (tất cả hyperparams)
│   ├── batch_test.py     # Vòng lặp evaluation
│   └── helper.py         # Early stopping, ensureDir
├── evaluator/
│   └── evaluate_foldout.py  # 6 metrics: recall/precision/ndcg/map/mrr/hit_ratio
└── scripts/              # Shell scripts train batch theo encoder/model
```

---

## Cách chạy

### Chạy 1 config

```bash
python3 train.py \
    --model bm3 \
    --data_path ../get10k_data/output_10k_sample \
    --dataset "" \
    --sim_type multimodal \
    --embed_type mbnv2 \
    --embed_size 512 \
    --layer_size "[512,512,512,512]" \
    --lr 0.001 \
    --regs "[1e-4]" \
    --batch_size 8192 \
    --epoch 1000 \
    --eval_interval 40 \
    --Ks "[1,5,10,20]" \
    --gpu_id 0
```

> `--data_path` trỏ tới `output_10k_sample` (MobileNetV2) hoặc `clip_10k_sample`
> (CLIP). `--embed_type` chỉ ảnh hưởng tên run/log — phải khớp với data_path.

### Chạy batch bằng scripts

Thư mục `scripts/` chứa script chạy toàn bộ variants theo từng (model, encoder):

| Script | Nội dung |
|---|---|
| `01_run_all_clip/` · `02_run_all_mbnv2/` | CombiGCN — 4 sim_types |
| `03_run_bm3_clip/` · `04_run_bm3_mbnv2/` | BM3 — 4 sim_types |
| `05_run_freedom_clip/` · `06_run_freedom_mbnv2/` | FREEDOM — 4 sim_types |
| `setup_env.sh` | Setup môi trường |

```bash
bash scripts/04_run_bm3_mbnv2/run_all.sh      # train cả 4 variants tuần tự
bash scripts/04_run_bm3_mbnv2/run_multimodal.sh   # chỉ 1 variant
```

---

## Tham số chính (cấu hình thực nghiệm)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `embed_size` | 512 | Kích thước embedding |
| `layer_size` | `[512,512,512,512]` | 4 layers GCN |
| `lr` | 0.001 | Learning rate |
| `batch_size` | 8192 | BPR batch |
| `epoch` | 1000 | Số epoch |
| `eval_interval` | 40 | Eval mỗi 40 epoch |
| `Ks` | `[1,5,10,20]` | Top-K đánh giá |
| `bm3_momentum` | 0.995 | (BM3) EMA momentum |
| `bm3_cl_weight` | 0.2 | (BM3) weight bootstrap CL |
| `freedom_knn_k` | 10 | (FREEDOM) số neighbors kNN |
| `freedom_cl_weight` | 0.1 | (FREEDOM) weight InfoNCE |
| `freedom_cl_temp` | 0.2 | (FREEDOM) temperature |

---

## Output & Logging

- **Metrics:** 6 metrics (recall, precision, ndcg, map, mrr, hit_ratio) @ K=1/5/10/20
- **Checkpoints:** `weights/<sim_type>/...` (best model theo recall)
- **Results:** `output/<sim_type>/<model>.result`
- **Logging:** TensorBoard (mặc định) + WandB (`--use_wandb 1`)
- **HuggingFace Hub:** push best model (`--use_hf 1 --hf_repo_id ...`)

---

## Liên quan

- Chuẩn bị dữ liệu: [`../get10k_data/`](../get10k_data/)
- Đánh giá & vẽ biểu đồ: [`../../data_evaluate/`](../../data_evaluate/)
