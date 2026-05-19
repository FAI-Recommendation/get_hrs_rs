# 06 — Scripts & Experiment Tracking

---

## Cấu trúc Scripts

```
rs/lightgcn_pyg/scripts/
│
├── test/                        # Test nhanh (20 epochs, verify pipeline)
│   ├── test_all.sh              # Chạy cả 4 variants liên tiếp
│   ├── test_lightgcn.sh         # LightGCN thuần (--sim_type none)
│   ├── test_img_only.sh         # CombiGCN + Image Only
│   ├── test_multimodal.sh       # CombiGCN + Multimodal
│   └── test_tfidf.sh            # CombiGCN + TF-IDF
│
├── run_all/                     # Full training (1000 epochs)
│   ├── run_all.sh               # Chạy cả 4 variants liên tiếp
│   ├── run_lightgcn.sh
│   ├── run_img_only.sh
│   ├── run_multimodal.sh
│   └── run_tfidf.sh
│
├── 01_run_all_clip/             # CombiGCN + CLIP, full run
├── 01_test_clip/                # CombiGCN + CLIP, test 20 epoch
├── 02_run_all_mbnv2/            # CombiGCN + MobileNetV2, full run
├── 02_test_mbnv2/
├── 03_run_bm3_clip/             # BM3 + CLIP
├── 03_test_bm3_clip/
├── 04_run_bm3_mbnv2/            # BM3 + MobileNetV2
├── 04_test_bm3_mbnv2/
├── 05_run_freedom_clip/         # FREEDOM + CLIP
├── 05_test_freedom_clip/
├── 06_run_freedom_mbnv2/        # FREEDOM + MobileNetV2
└── 06_test_freedom_mbnv2/
```

---

## Cách chạy

```bash
# Full run (1000 epochs)
bash rs/lightgcn_pyg/scripts/<folder>/run_all.sh

# Chạy từng sim_type
bash rs/lightgcn_pyg/scripts/<folder>/run_img_only.sh
bash rs/lightgcn_pyg/scripts/<folder>/run_tfidf.sh
bash rs/lightgcn_pyg/scripts/<folder>/run_multimodal.sh
bash rs/lightgcn_pyg/scripts/<folder>/run_multimodal_attention.sh

# Test pipeline nhanh (20 epochs)
bash rs/lightgcn_pyg/scripts/<folder>/test_all.sh
```

---

## Tham số — Test Scripts (20 epochs)

| Tham số | Giá trị | Lý do |
|---|---|---|
| `--epoch` | 20 | Chỉ verify pipeline, không cần hội tụ |
| `--eval_interval` | 10 | Eval 2 lần: epoch 10 và 20 |
| `--early_stop_steps` | 3 | Dừng sớm nếu bị lỗi |
| `--Ks` | [1,5,10,20] | Bỏ K=50 để giảm thời gian eval |
| `--save_flag` | 0 | Không lưu weights |
| `--batch_size` | 1024 | Giống full training |

---

## Tham số — Full Training Scripts (1000 epochs)

| Tham số | Giá trị | Lý do |
|---|---|---|
| `--epoch` | 1000 | Đủ epochs để hội tụ |
| `--eval_interval` | 40 | Eval mỗi 40 epochs |
| `--early_stop_steps` | 0 | Chạy hết (không early stop) |
| `--Ks` | [1,5,10,20] | Đầy đủ K values |
| `--save_flag` | 1 | Lưu best model |
| `--batch_size` | 8192 | Tận dụng GPU lớn |

---

## So sánh 4 Variants (CombiGCN)

| Variant | `--sim_type` | File TF1 gốc | Đặc điểm |
|---|---|---|---|
| LightGCN thuần | `none` | `LightGCN.py` | Chỉ bipartite graph, nhanh nhất |
| CombiGCN + Image | `img_only` | `LightGCN_only_img.py` | Thêm image similarity giữa items |
| CombiGCN + Multimodal | `multimodal` | `LightGCN_bert_img.py` | alpha*text + (1-alpha)*image |
| CombiGCN + TF-IDF | `tfidf` | `LightGCN_tfidf_bert.py` | TF-IDF text similarity giữa items |

Thứ tự nên chạy: `none` → `img_only` → `multimodal` → `tfidf`

---

## Tracking 24 Thực Nghiệm

| # | Script folder | Model | Embedding | sim_type | Status |
|---|---------------|-------|-----------|----------|--------|
| 1 | `01_run_all_clip` | CombiGCN | CLIP | img_only | ⬜ |
| 2 | `01_run_all_clip` | CombiGCN | CLIP | tfidf | ⬜ |
| 3 | `01_run_all_clip` | CombiGCN | CLIP | multimodal | ⬜ |
| 4 | `01_run_all_clip` | CombiGCN | CLIP | multimodal_attention | ⬜ |
| 5 | `02_run_all_mbnv2` | CombiGCN | MobileNetV2 | img_only | ⬜ |
| 6 | `02_run_all_mbnv2` | CombiGCN | MobileNetV2 | tfidf | ⬜ |
| 7 | `02_run_all_mbnv2` | CombiGCN | MobileNetV2 | multimodal | ⬜ |
| 8 | `02_run_all_mbnv2` | CombiGCN | MobileNetV2 | multimodal_attention | ⬜ |
| 9 | `03_run_bm3_clip` | BM3 | CLIP | img_only | 🔄 |
| 10 | `03_run_bm3_clip` | BM3 | CLIP | tfidf | ⬜ |
| 11 | `03_run_bm3_clip` | BM3 | CLIP | multimodal | ⬜ |
| 12 | `03_run_bm3_clip` | BM3 | CLIP | multimodal_attention | ⬜ |
| 13 | `04_run_bm3_mbnv2` | BM3 | MobileNetV2 | img_only | ⬜ |
| 14 | `04_run_bm3_mbnv2` | BM3 | MobileNetV2 | tfidf | ⬜ |
| 15 | `04_run_bm3_mbnv2` | BM3 | MobileNetV2 | multimodal | ⬜ |
| 16 | `04_run_bm3_mbnv2` | BM3 | MobileNetV2 | multimodal_attention | ⬜ |
| 17 | `05_run_freedom_clip` | FREEDOM | CLIP | img_only | ⬜ |
| 18 | `05_run_freedom_clip` | FREEDOM | CLIP | tfidf | ⬜ |
| 19 | `05_run_freedom_clip` | FREEDOM | CLIP | multimodal | ⬜ |
| 20 | `05_run_freedom_clip` | FREEDOM | CLIP | multimodal_attention | ⬜ |
| 21 | `06_run_freedom_mbnv2` | FREEDOM | MobileNetV2 | img_only | ⬜ |
| 22 | `06_run_freedom_mbnv2` | FREEDOM | MobileNetV2 | tfidf | ⬜ |
| 23 | `06_run_freedom_mbnv2` | FREEDOM | MobileNetV2 | multimodal | ⬜ |
| 24 | `06_run_freedom_mbnv2` | FREEDOM | MobileNetV2 | multimodal_attention | ⬜ |

**Legend:** ⬜ chờ &nbsp;|&nbsp; 🔄 đang chạy &nbsp;|&nbsp; ✅ xong &nbsp;|&nbsp; ❌ lỗi

---

## Output files

```
rs/lightgcn_pyg/output/<model>_<sim_type>_<embed_type>/
├── <model>.result          ← Recall@K, NDCG@K từng epoch
└── ...

rs/lightgcn_pyg/weights/<model>_<sim_type>_<embed_type>/
└── <model>_epoch<N>.pth    ← checkpoint mỗi 200 epoch
```

---

## Tùy chỉnh tham số

### Thay đổi data path

```bash
--data_path ../../get10k_data/clip_10k_sample    # CLIP (mặc định)
--data_path ../../get10k_data/mbnv2_10k_sample   # MobileNetV2
```

### Thay đổi GPU

```bash
--gpu_id 0    # GPU đầu tiên (mặc định)
--gpu_id 1    # GPU thứ hai
--gpu_id -1   # CPU (chậm, chỉ để test)
```

### Tăng batch size

```bash
--batch_size 2048    # ~4GB VRAM
--batch_size 4096    # ~8GB VRAM
--batch_size 8192    # ~16GB VRAM
```

### Thay đổi số layers

```bash
--layer_size "[64,64]"          # 2 layers
--layer_size "[64,64,64]"       # 3 layers (mặc định nhẹ)
--layer_size "[512,512,512,512]" # 4 layers (full run)
```

---

## TensorBoard

```bash
tensorboard --logdir rs/lightgcn_pyg/tensorboard/
```

Mở browser tại `http://localhost:6006` để xem:
- **Scalars**: loss, recall, precision, ndcg, ... theo epoch
- **So sánh**: chọn nhiều runs để so sánh các variants
