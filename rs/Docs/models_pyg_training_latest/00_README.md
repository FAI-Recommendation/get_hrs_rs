# models_pyg_training_latest — Tổng Quan

Tài liệu hợp nhất từ 3 folder cũ:
- `lightgcn_pyg/` — CombiGCN (PyG implementation)
- `model_data_emb/` — 3 models × 2 embeddings × 24 thực nghiệm
- `data_raw_toEnd/` — Pipeline từ raw VCR → train/test

---

## Cấu trúc tài liệu

| File | Nội dung |
|------|----------|
| `01_data_raw_pipeline.md` | Raw VCR → 10k sample → train/test |
| `02_model_architectures.md` | CombiGCN / BM3 / FREEDOM — kiến trúc & input/output |
| `03_data_embeddings.md` | CLIP, MobileNetV2, adjacency matrices, data pipeline cho model |
| `04_training_pipeline.md` | Loss, optimizer, early stopping, hyperparameters |
| `05_evaluation.md` | 6 metrics + glossary |
| `06_scripts_experiments.md` | Scripts + tracking 24 thực nghiệm |
| `07_remote_gpu.md` | Remote GPU workflow (RunPod / Vast.ai) |
| `08_troubleshooting.md` | Lỗi thực tế và cách fix |

---

## Ma trận thực nghiệm

| # | Model | Embedding | Data folder | sim_type |
|---|-------|-----------|-------------|----------|
| 1 | CombiGCN | CLIP | `clip_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 2 | CombiGCN | MobileNetV2 | `mbnv2_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 3 | BM3 | CLIP | `clip_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 4 | BM3 | MobileNetV2 | `mbnv2_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 5 | FREEDOM | CLIP | `clip_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 6 | FREEDOM | MobileNetV2 | `mbnv2_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |

**Tổng: 3 models × 2 embeddings × 4 sim_type = 24 thực nghiệm**

> **Lưu ý về multimodal:** Trong 4 sim_type, có 2 sim_type là multimodal — mỗi cái tương ứng với 1 fusion strategy khác nhau (xem bảng bên dưới). Đây là 2 phương pháp chính đang được train và evaluate.

---

## Hai Multimodal Fusion Method đang Train & Evaluate

Khi `sim_type` thuộc nhóm multimodal, model kết hợp **image embedding + text embedding** theo 2 chiến lược:

| sim_type | Fusion method | Cách hoạt động | Config |
|---|---|---|---|
| `multimodal` | **Late Fusion** (default) | Tính cosine similarity riêng cho image và text, rồi gộp: `alpha * text_sim + (1-alpha) * img_sim` với `alpha=0.5` | `MULTIMODAL_METHOD=late_fusion` |
| `multimodal_attention` | **Weight Attention Fusion** | Học trọng số tự động qua MultiHeadAttention — model tự quyết định khi nào tin image hơn, khi nào tin text hơn | `MULTIMODAL_METHOD=attention` |

**Tại sao chọn 2 method này?**
- **Late Fusion** — baseline ổn định, không cần train thêm tham số, cho kết quả tốt hơn early fusion rõ rệt
- **Weight Attention** — học trọng số từ data, đạt kết quả tốt nhất trong các phương pháp multimodal

Hai method còn lại (`aggregation`, `pca`) đã được thử nghiệm sơ bộ nhưng không đưa vào bộ 24 thực nghiệm chính vì kết quả không vượt trội. Xem chi tiết tại [`06_ket_qua_mo_hinh.md`](../06_ket_qua_mo_hinh.md).

---

## CombiGCN — Mapping TF1 → PyG

CombiGCN chuyển đổi từ **TensorFlow 1.x** sang **PyTorch Geometric (PyG)**, gom 4 file gốc TF1 thành **1 class duy nhất** với tham số `--sim_type`.

| File gốc (TF1) | Dòng code | Code mới (PyG) | Cách gọi |
|---|---|---|---|
| `hr/LightGCN.py` | ~650 | `train.py --sim_type none` | LightGCN thuần |
| `hr/LightGCN_bert_img.py` | ~650 | `train.py --sim_type multimodal` | BERT text + Image |
| `hr/LightGCN_only_img.py` | ~650 | `train.py --sim_type img_only` | Image only |
| `hr/LightGCN_tfidf_bert.py` | ~650 | `train.py --sim_type tfidf` | TF-IDF text |

**Gốc: ~8 files, ~2800+ dòng** → **Mới: 5 files, ~900 dòng** (giảm ~68%)

---

## Cấu trúc thư mục code

```
rs/lightgcn_pyg/
├── train.py                        # Entry point — import từ models/, gọi utility/
├── model.py                        # (legacy) CombiGCN gốc, không còn được import
│
├── models/                         # Package chứa 3 model
│   ├── __init__.py                 # Export: CombiGCN, BM3, FREEDOM, scipy_to_sparse_tensor
│   ├── combigcn.py                 # CombiGCN — dual-graph GCN
│   ├── bm3.py                      # BM3 — bootstrap multimodal
│   └── freedom.py                  # FREEDOM — frozen kNN + InfoNCE
│
├── utility/
│   ├── load_data.py                # Data loading + 8 adjacency matrices + BPR sampling
│   ├── batch_test.py               # Evaluation loop (foldout, mask train items)
│   ├── helper.py                   # early_stopping, ensureDir
│   └── parser.py                   # Argument parser (--sim_type, --model, ...)
│
├── evaluator/
│   └── evaluate_foldout.py         # Tính 6 metrics tích lũy
│
└── scripts/
    ├── setup_env.sh                # Cài môi trường tự động (CUDA detect, PyG wheels)
    │
    ├── 01_run_all_clip/            # CombiGCN × CLIP — full training
    │   ├── run_all.sh
    │   ├── run_lightgcn.sh
    │   ├── run_img_only.sh
    │   ├── run_tfidf.sh
    │   ├── run_multimodal.sh
    │   └── run_multimodal_attention.sh
    ├── 01_test_clip/               # CombiGCN × CLIP — smoke test (20 epochs)
    │
    ├── 02_run_all_mbnv2/           # CombiGCN × MobileNetV2 — full training
    ├── 02_test_mbnv2/              # CombiGCN × MobileNetV2 — smoke test
    │
    ├── 03_run_bm3_clip/            # BM3 × CLIP — full training
    ├── 03_test_bm3_clip/           # BM3 × CLIP — smoke test
    │
    ├── 04_run_bm3_mbnv2/           # BM3 × MobileNetV2 — full training
    ├── 04_test_bm3_mbnv2/          # BM3 × MobileNetV2 — smoke test
    │
    ├── 05_run_freedom_clip/        # FREEDOM × CLIP — full training
    ├── 05_test_freedom_clip/       # FREEDOM × CLIP — smoke test
    │
    ├── 06_run_freedom_mbnv2/       # FREEDOM × MobileNetV2 — full training
    └── 06_test_freedom_mbnv2/      # FREEDOM × MobileNetV2 — smoke test
```

> `train.py` import model qua `from models import CombiGCN, BM3, FREEDOM` — `model.py` ở root là file cũ, giữ lại để tham khảo nhưng không còn được sử dụng.

---

## Quick Start

```bash
# 1. Cài dependencies
pip install torch torch-geometric torch-sparse tensorboard transformers

# 2. Test nhanh (20 epochs)
cd rs/lightgcn_pyg
bash scripts/test/test_all.sh

# 3. Full training
bash scripts/run_all/run_all.sh
```

---

## Tham số chung (full run)

| Tham số | Giá trị |
|---------|---------|
| `embed_size` | 512 |
| `layer_size` | [512, 512, 512, 512] |
| `lr` | 0.001 |
| `regs` | [1e-4] |
| `batch_size` | 8192 |
| `epoch` | 1000 |
| `eval_interval` | 40 |
| `early_stop_steps` | 0 (chạy hết) |
| `Ks` | [1, 5, 10, 20] |
