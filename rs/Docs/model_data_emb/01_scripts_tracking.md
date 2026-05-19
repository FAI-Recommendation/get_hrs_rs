# 01 – Script Tracking: Các thực nghiệm đang chạy

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

## Bảng tracking

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

Sau khi chạy xong, kết quả lưu tại:

```
rs/lightgcn_pyg/output/<model>_<sim_type>_<embed_type>/
├── <model>.result          ← Recall@K, NDCG@K từng epoch
└── ...

rs/lightgcn_pyg/weights/<model>_<sim_type>_<embed_type>/
└── <model>_epoch<N>.pth    ← checkpoint mỗi 200 epoch
```
