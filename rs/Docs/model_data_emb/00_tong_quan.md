# 00 – Tổng Quan: Models × Embeddings × Data

## Danh sách thực nghiệm

| # | Model | Embedding | Data folder | sim_type |
|---|-------|-----------|-------------|----------|
| 1 | CombiGCN | CLIP | `clip_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 2 | CombiGCN | MobileNetV2 | `mbnv2_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 3 | BM3 | CLIP | `clip_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 4 | BM3 | MobileNetV2 | `mbnv2_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 5 | FREEDOM | CLIP | `clip_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |
| 6 | FREEDOM | MobileNetV2 | `mbnv2_10k_sample` | img_only / tfidf / multimodal / multimodal_attention |

**Tổng: 3 models × 2 embeddings × 4 sim_type = 24 thực nghiệm**

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

---

## Script folders

| Folder | Model | Embedding | Mô tả |
|--------|-------|-----------|-------|
| `01_run_all_clip/` | CombiGCN | CLIP | Full run, 4 sim_type |
| `01_test_clip/` | CombiGCN | CLIP | Test 20 epoch |
| `02_run_all_mbnv2/` | CombiGCN | MobileNetV2 | Full run, 4 sim_type |
| `02_test_mbnv2/` | CombiGCN | MobileNetV2 | Test 20 epoch |
| `03_run_bm3_clip/` | BM3 | CLIP | Full run, 4 sim_type |
| `03_test_bm3_clip/` | BM3 | CLIP | Test 20 epoch |
| `04_run_bm3_mbnv2/` | BM3 | MobileNetV2 | Full run, 4 sim_type |
| `04_test_bm3_mbnv2/` | BM3 | MobileNetV2 | Test 20 epoch |
| `05_run_freedom_clip/` | FREEDOM | CLIP | Full run, 4 sim_type |
| `05_test_freedom_clip/` | FREEDOM | CLIP | Test 20 epoch |
| `06_run_freedom_mbnv2/` | FREEDOM | MobileNetV2 | Full run, 4 sim_type |
| `06_test_freedom_mbnv2/` | FREEDOM | MobileNetV2 | Test 20 epoch |

Xem chi tiết: `01_models.md`, `02_embeddings.md`, `03_data.md`
