# 01 — Tong quan Pipeline & Kien truc he thong

## Cau truc thu muc

```
lightgcn_pyg/
├── train.py                  # Entry point duy nhat (train + eval)
├── model.py                  # CombiGCN legacy (deprecated, dung combigcn.py)
├── models/
│   ├── __init__.py           # Export: CombiGCN, BM3, FREEDOM, scipy_to_sparse_tensor
│   ├── combigcn.py           # CombiGCN / LightGCN thuan
│   ├── bm3.py                # BM3 (Bootstrap Multimodal)
│   └── freedom.py            # FREEDOM (Freeze & Denoise)
├── utility/
│   ├── parser.py             # Argument parser (chung 3 models)
│   ├── load_data.py          # Data loader + similarity matrices + BPR sampling
│   ├── batch_test.py         # Evaluation loop (chung 3 models)
│   └── helper.py             # early_stopping, ensureDir
├── evaluator/
│   └── evaluate_foldout.py   # 6 metrics: precision, recall, map, ndcg, mrr, hit_ratio
└── scripts/
    ├── 01_run_all_clip/      # CombiGCN + CLIP
    ├── 02_run_all_mbnv2/     # CombiGCN + MBNv2
    ├── 03_run_bm3_clip/      # BM3 + CLIP
    ├── 04_run_bm3_mbnv2/     # BM3 + MBNv2
    ├── 05_run_freedom_clip/  # FREEDOM + CLIP
    └── 06_run_freedom_mbnv2/ # FREEDOM + MBNv2
```

## Pipeline tong the (chung ca 3 models)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         train.py (main)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. parse_args()         ← parser.py                                │
│  2. Data(path, batch)    ← load_data.py                             │
│     ├─ doc train.txt / test.txt                                     │
│     ├─ build R matrix (user×item sparse)                            │
│     └─ doc items_features.csv                                       │
│                                                                     │
│  3. Build adjacency matrices                                        │
│     ├─ CombiGCN: interaction_adj + similarity_adj (theo sim_type)   │
│     └─ BM3/FREEDOM: interaction_adj only + raw embeddings (.npy)    │
│                                                                     │
│  4. Create model                                                    │
│     ├─ CombiGCN(n_users, n_items, dim, layers, decay, dropout)      │
│     ├─ BM3(..., image_feats, text_feats, momentum, cl_weight)       │
│     └─ FREEDOM(..., image_feats, text_feats, knn_k, cl_weight)      │
│                                                                     │
│  5. Training loop (1000 epochs)                                     │
│     ├─ BPR sampling: (user, pos_item, neg_item)                     │
│     ├─ Forward → loss = bpr + reg [+ contrastive]                   │
│     ├─ AMP fp16 (neu co GPU)                                        │
│     ├─ Log: TensorBoard + WandB                                     │
│     └─ Eval moi N epoch → test() → early_stopping                  │
│                                                                     │
│  6. Save: best_model.pt, checkpoints, best_metrics.json             │
│  7. Push: HuggingFace Hub (optional)                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 3 models — so sanh nhanh

| Dac diem | CombiGCN | BM3 | FREEDOM |
|---|---|---|---|
| **Paper** | Custom (GCN dual-graph) | Zhou et al., WWW 2023 | Zhou et al., ACM MM 2023 |
| **Input features** | Similarity matrix (scipy) | Raw embeddings (.npy) | Raw embeddings (.npy) |
| **Graph** | interaction + item-sim (precomputed) | interaction only | interaction + kNN item graph (built at init) |
| **Loss** | BPR + L2 reg | BPR + L2 + Bootstrap CL | BPR + L2 + InfoNCE CL |
| **Multimodal fusion** | Tai Data layer (similarity matrix) | Tai Model layer (projector) | Tai Model layer (projector + kNN graph) |
| **sim_type** | none/img_only/tfidf/multimodal | img_only/tfidf/multimodal/mm_attention | img_only/tfidf/multimodal/mm_attention |

## Hyperparameters chung (config thuc nghiem)

```
embed_size    = 512
layer_size    = [512, 512, 512, 512]   # 4 layers
lr            = 0.001
regs          = [1e-4]
batch_size    = 8192
epoch         = 1000
eval_interval = 40
Ks            = [1, 5, 10, 20]
```

## Luong du lieu (Data flow)

```
Raw VCR (~64k) → sample 10k → N-Core 5 filter → temporal split 80/20
                                                        │
                              ┌─────────────────────────┼────────────────────┐
                              │                         │                    │
                        train.txt               test.txt           items_features.csv
                   (uid item1 item2 ...)     (uid item1 ...)    (feature1, feature2, feature3)
                              │                                         │
                              │                              ┌──────────┴───────────┐
                              │                              │                      │
                         R matrix              image_embeddings.npy    text_embeddings.npy
                    (user×item sparse)         (CLIP hoac MBNv2)       (TF-IDF vectors)
                              │
                    ┌─────────┴──────────┐
                    │                    │
             interaction_adj       similarity matrices
           (bipartite graph       (tfidf/img/multimodal)
            D^-0.5 A D^-0.5)         (CombiGCN only)
```

## Evaluation pipeline

```
test() ← batch_test.py
  │
  ├─ model.predict(interaction_adj, sim_adj, users)  → scores [batch, n_items]
  ├─ Mask training items → -inf
  ├─ argmax_top_k(scores, max_K)                     → ranking
  └─ eval_score_matrix_foldout(scores, test_items)
       ├─ precision@K    (cumulative hits / K)
       ├─ recall@K       (cumulative hits / |ground_truth|)
       ├─ ndcg@K         (DCG / IDCG)
       ├─ map@K          (mean average precision)
       ├─ mrr@K          (reciprocal rank)
       └─ hit_ratio@K    (co it nhat 1 hit?)
```
