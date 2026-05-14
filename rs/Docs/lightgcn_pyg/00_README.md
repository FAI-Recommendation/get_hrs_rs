# CombiGCN — PyG Implementation

## Tổng quan

CombiGCN (Combining Graph Convolution Network) la mo hinh goi y (recommendation) dua tren do thi, ket hop LightGCN voi thong tin tuong tu giua cac item (van ban, hinh anh, da phuong thuc).

Project nay chuyen doi tu **TensorFlow 1.x** sang **PyTorch Geometric (PyG)**, gom 4 file goc TF1 thanh **1 class duy nhat** voi tham so `--sim_type`.

## Mapping: Code goc → Code moi

| File goc (TF1) | Dong code | Code moi (PyG) | Cach goi |
|---|---|---|---|
| `hr/LightGCN.py` | ~650 | `train.py --sim_type none` | LightGCN thuan |
| `hr/LightGCN_bert_img.py` | ~650 | `train.py --sim_type multimodal` | BERT text + Image |
| `hr/LightGCN_only_img.py` | ~650 | `train.py --sim_type img_only` | Image only |
| `hr/LightGCN_tfidf_bert.py` | ~650 | `train.py --sim_type tfidf` | TF-IDF text |
| `hr/utility/load_data.py` | ~636 | `utility/load_data.py` | Bo TF, them PyG convert |
| `hr/utility/batch_test.py` | ~94 | `utility/batch_test.py` | Bo TF Session |
| `hr/utility/helper.py` | ~63 | `utility/helper.py` | Copy nguyen |
| `hr/utility/parser.py` | ~64 | `utility/parser.py` | Them --sim_type |
| `hr/evaluator/python/evaluate_foldout.py` | ~80 | `evaluator/evaluate_foldout.py` | Copy nguyen |

**Goc: ~8 files, ~2800+ dong** → **Moi: 5 files, ~900 dong** (giam ~68%)

## Cau truc thu muc

```
rs/lightgcn_pyg/
├── model.py                    # CombiGCN model (4 variants)
├── train.py                    # Training entry point
├── utility/
│   ├── load_data.py            # Data loading + adjacency matrices
│   ├── batch_test.py           # Evaluation loop
│   ├── helper.py               # early_stopping, ensureDir
│   └── parser.py               # Argument parser
├── evaluator/
│   └── evaluate_foldout.py     # 6 metrics calculation
└── scripts/
    ├── test/                   # Quick test (20 epochs)
    └── run_all/                # Full training (1000 epochs)
```

## Quick Start

```bash
# 1. Cai dependencies
pip install torch torch-geometric torch-sparse tensorboard transformers

# 2. Test nhanh (20 epochs)
cd rs/lightgcn_pyg
bash scripts/test/test_all.sh

# 3. Full training
bash scripts/run_all/run_all.sh
```

## Tai lieu chi tiet

- [ARCHITECTURE.md](ARCHITECTURE.md) — Kien truc model, dual-graph propagation
- [TRAINING.md](TRAINING.md) — Training pipeline, loss, optimizer, early stopping
- [EVALUATION.md](EVALUATION.md) — 6 metrics, cach tinh, data flow
- [DATA_PIPELINE.md](DATA_PIPELINE.md) — Data loading, adjacency matrices, sampling
- [SCRIPTS.md](SCRIPTS.md) — Huong dan chay scripts test va run_all
- [REMOTE_GPU.md](REMOTE_GPU.md) — Thue GPU cloud (RunPod/Vast.ai), setup SSH, cai dependencies
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Cac loi thuc te va cach fix
