# get10k_data — Data Pipeline

Pipeline chuẩn bị dữ liệu cho hệ thống gợi ý thời trang đa phương thức: từ raw
VCR (~64k giao dịch) → sample 10k → lọc → chia train/test → sinh đặc trưng
(image + text embeddings) → build các ma trận adjacency.

---

## Luồng xử lý

```
raw VCR (~64k giao dịch)
        │
        ▼
[1] slipt_10k_sample.py        → sample ~10k giao dịch
        │
        ▼
[2] preprocess_vcr_*.py        → N-Core 5 filter, temporal split 80/20
        │                         build train.txt / test.txt + adjacency .npz
        ▼
[3] get_embedding_*.py         → trích image embeddings (.npy)
    gen_embeddings_mbnv2.py       + text embeddings (BERT/TF-IDF)
        │
        ▼
   output_10k_sample/  (MobileNetV2)
   clip_10k_sample/    (CLIP)
```

---

## Mô tả các file

| File | Vai trò |
|---|---|
| `slipt_10k_sample.py` | Sample ~10k giao dịch từ raw VCR |
| `copy_images_10k.py` | Copy ảnh chính (`displayOrder == 0`) của các item trong sample |
| `preprocess_vcr_10k.py` | Pipeline đầy đủ cho nhánh **MobileNetV2 + BERT**: N-Core filter, split, build adjacency |
| `preprocess_vcr_clip.py` | Pipeline cho nhánh **CLIP** (image 512-D, không cần PCA) |
| `get_embedding_CLIP.py` | Trích image embedding bằng CLIP ViT-B/32 → vector 512 chiều |
| `get_embedding_MBNV2_optimized.py` | Trích image embedding bằng MobileNetV2 (bản tối ưu) |
| `gen_embeddings_mbnv2.py` | Sinh `image_embeddings.npy` từ MobileNetV2 |
| `data_processing_and_embedding.ipynb` | Notebook tổng hợp toàn bộ pipeline (khám phá + chạy thử) |
| `preprocess/` | Dữ liệu trung gian: `outfits.csv`, `picture_triplets.csv`, `user_activity_triplets.csv` |

---

## Output

Hai thư mục output tương ứng 2 encoder ảnh:

| Thư mục | Encoder ảnh | Text |
|---|---|---|
| `output_10k_sample/` | MobileNetV2 | BERT (768-D) |
| `clip_10k_sample/` | CLIP ViT-B/32 (512-D) | BERT (768-D) |

Mỗi thư mục chứa:

| File | Nội dung |
|---|---|
| `train.txt` / `test.txt` | Interaction list (user → items), split temporal 80/20 |
| `user_list.txt` / `item_list.txt` | Mapping ID ↔ original ID |
| `items_features.csv` | Đặc trưng item (text + đường dẫn ảnh) |
| `image_embeddings.npy` | Image embeddings `[n_items, img_dim]` |
| `text_embeddings.npy` | Text embeddings `[n_items, 768]` (BERT) |
| `s_interaction_adj_mat.npz` | Bipartite user-item graph (cho LightGCN/CF) |
| `s_img_similarity_adj_mat.npz` | Item-item similarity từ ảnh |
| `s_tfidf_item_similarity_adj_mat.npz` | Item-item similarity từ text (TF-IDF) |
| `s_multimodal_late_fusion_similarity_adj_mat.npz` | Multimodal — late fusion |
| `s_multimodal_attention_similarity_adj_mat.npz` | Multimodal — weight attention |

> Các file `.npz` là **similarity matrix precompute sẵn** dùng cho CombiGCN.
> BM3/FREEDOM dùng trực tiếp `image_embeddings.npy` + `text_embeddings.npy`
> (build graph/projection ngay trong model).

---

## Cách chạy

```bash
# 1. Sample 10k giao dịch
python slipt_10k_sample.py

# 2a. Pipeline MobileNetV2 + BERT  → output_10k_sample/
python preprocess_vcr_10k.py

# 2b. Pipeline CLIP + BERT  → clip_10k_sample/
python preprocess_vcr_clip.py

# 3. (nếu cần regenerate) sinh image embeddings riêng
python get_embedding_CLIP.py
python gen_embeddings_mbnv2.py
```

---

## Thống kê dataset

| Chỉ số | Giá trị |
|---|---|
| Users | 553 |
| Items | 2.194 |
| Train interactions | 7.350 |
| Test interactions | 2.105 |
| Trung bình test items/user | ~3.8 |
| Lọc | N-Core 5, ảnh chính `displayOrder == 0` |

---

## Liên quan

- Train models trên dataset này: [`../lightgcn_pyg/`](../lightgcn_pyg/)
- Đánh giá kết quả: [`../../data_evaluate/`](../../data_evaluate/)
