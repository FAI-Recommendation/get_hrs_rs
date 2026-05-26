<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Bạn dựa vào đây kiếm link paper cho mình được không

# 02 – Input / Output của từng Model

## Dữ liệu đầu vào chung

Tất cả model đều đọc từ cùng 1 data folder (ví dụ `../get10k_data/clip_10k_sample`):


| File | Nội dung | Shape |
| :-- | :-- | :-- |
| `train.txt` | User-item interactions (training) | `n_interactions` rows |
| `test.txt` | User-item interactions (test) | `n_interactions` rows |
| `image_embeddings.npy` | Image features của items | `(n_items, img_dim)` |
| `text_embeddings.npy` | Text features của items (TF-IDF) | `(n_items, txt_dim)` |
| `s_interaction_adj_mat.npz` | Normalized user-item graph (cache) | `(n_users+n_items, n_users+n_items)` |


---

## CombiGCN

**Paper:** Dual-graph GCN kết hợp interaction graph + item similarity graph

```
INPUT
├── interaction_adj     SparseTensor (n_users+n_items, n_users+n_items)  ← user-item graph
├── similarity_adj      SparseTensor (n_items, n_items)                  ← item-item sim graph
├── image_feats         Tensor (n_items, img_dim)   [nếu sim_type dùng image]
├── text_feats          Tensor (n_items, txt_dim)   [nếu sim_type dùng text]
├── users               Tensor (batch,)
├── pos_items           Tensor (batch,)
└── neg_items           Tensor (batch,)


OUTPUT (forward)
├── loss        scalar  ← BPR + reg
├── bpr_loss    scalar
└── reg_loss    scalar


OUTPUT (predict)
└── scores      Tensor (batch_users, n_items)  ← ranking scores
```

**sim_type ảnh hưởng đến:**

- `img_only` → similarity_adj từ image cosine similarity
- `tfidf` → similarity_adj từ TF-IDF text similarity
- `multimodal` → similarity_adj = avg(img_sim, txt_sim)
- `multimodal_attention` → similarity_adj học trọng số attention

---

## BM3

**Paper:** Bootstrap Latent Representations for Multi-modal Recommendation, WWW 2023

```
INPUT
├── interaction_adj     SparseTensor (n_users+n_items, n_users+n_items)
├── image_feats         Tensor (n_items, img_dim)   [nếu sim_type != tfidf]
├── text_feats          Tensor (n_items, txt_dim)   [nếu sim_type != img_only]
├── users               Tensor (batch,)
├── pos_items           Tensor (batch,)
└── neg_items           Tensor (batch,)


OUTPUT (forward)
├── loss        scalar  ← BPR + reg + cl_weight × bootstrap_cl_loss
├── bpr_loss    scalar
└── reg_loss    scalar


OUTPUT (predict)
└── scores      Tensor (batch_users, n_items)
```

**Kiến trúc bên trong:**

```
user/item ID embeddings
        │
   LightGCN propagation (interaction_adj)
        │
  item_emb_cf ──────────────────────────────────────────┐
        │                                               │
  modal projector(s)                            EMA target encoder
  image_projector / text_projector / attention_fusion   │
        │                                               │
  item_emb_modal ──── Bootstrap CL loss ───── item_emb_target
        │
  item_emb = item_emb_cf + item_emb_modal
```

**sim_type ảnh hưởng đến modal fusion:**

- `img_only` → `image_projector(image_feats)`
- `tfidf` → `text_projector(text_feats)`
- `multimodal` → `(img + txt) / 2`
- `multimodal_attention` → `Linear(concat(img, txt))`

---

## FREEDOM

**Paper:** Freezing and Denoising Graph Structures for Multimodal Recommendation, ACM MM 2023

```
INPUT
├── interaction_adj     SparseTensor (n_users+n_items, n_users+n_items)
├── image_feats         Tensor (n_items, img_dim)   [nếu sim_type != tfidf]
├── text_feats          Tensor (n_items, txt_dim)   [nếu sim_type != img_only]
├── users               Tensor (batch,)
├── pos_items           Tensor (batch,)
└── neg_items           Tensor (batch,)


OUTPUT (forward)
├── loss        scalar  ← BPR + reg + cl_weight × InfoNCE
├── bpr_loss    scalar
└── reg_loss    scalar


OUTPUT (predict)
└── scores      Tensor (batch_users, n_items)
```

**Kiến trúc bên trong:**

```
modal features (image / text / fused)
        │
  build_knn_item_graph()  ← tính 1 lần lúc __init__, FROZEN
        │
  frozen item-item graph (SparseTensor)
        │
  content propagation ──── item_emb_content
                                  │
user/item ID embeddings           │ InfoNCE loss
        │                         │
  LightGCN propagation ── item_emb_cf
        │
  item_emb = item_emb_cf + item_emb_content
```

**Điểm khác biệt với BM3:**

- BM3: modal features → projector → cộng vào CF embedding mỗi forward
- FREEDOM: modal features → build kNN graph 1 lần → propagate trên frozen graph → InfoNCE với CF view

**sim_type ảnh hưởng đến:**

- Cách build kNN item graph (img_only / tfidf / avg của 2 modality)
- Cách tính modal embedding cho content propagation

---

## So sánh nhanh

|  | CombiGCN | BM3 | FREEDOM |
| :-- | :-- | :-- | :-- |
| Item-item graph | Similarity matrix (cosine/attention) | Không có | Frozen kNN graph từ modal |
| Modal dùng cho | Build similarity graph | Projector + bootstrap CL | Build kNN graph + content propagation |
| Contrastive loss | Không | Bootstrap (online ↔ target) | InfoNCE (CF view ↔ content view) |
| Pure CF mode | ✅ (sim_type=none) | ❌ | ❌ |

Được. Với đoạn bạn đưa, mình có thể chỉ ra **4 paper chính** tương ứng, kèm link chuẩn để bạn cite vào phần mô tả Input/Output của từng model.[^1][^2][^3][^4]

## Link paper theo từng model

### CombiGCN

Paper phù hợp nhất là **CombiGCN: An effective GCN model for Recommender System**.[^1]
Link: [arXiv](https://arxiv.org/abs/2503.21471) hoặc [DOI/Springer](https://doi.org/10.1007/978-981-97-0669-3_11).[^5][^1]
Điểm khớp với mô tả của bạn là CombiGCN dùng **user-item interaction graph** và thêm **user-user weighted connection graph**; nó không hoàn toàn là item-item similarity graph như bạn đang diễn giải, nên nếu bạn viết luận văn thì nên chỉnh wording cho sát paper gốc hơn.[^5][^1]

### BM3

Paper phù hợp nhất là **Bootstrap Latent Representations for Multi-modal Recommendation**.[^2]
Link: [arXiv](https://arxiv.org/abs/2207.05969).[^2]
Paper này đúng với mô tả của bạn ở chỗ BM3 là mô hình multimodal recommendation, dùng bootstrap latent contrastive views và không cần auxiliary graphs hay negative samples.[^2]

### FREEDOM

Paper chính là **A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation**.[^4]
Link: [arXiv](https://arxiv.org/abs/2211.06924) hoặc [ACM DL](https://dl.acm.org/doi/10.1145/3581783.3611943).[^3][^4]
Mô tả của bạn về “frozen item-item graph” và “denoising user-item graph” khớp rất sát với paper gốc.[^4]

### LightGCN

Paper nền tảng là **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**.[^1][^2]
Link: [arXiv](https://arxiv.org/abs/2002.02126).[^1]
Nếu bạn dùng LightGCN làm baseline trong phần so sánh, paper này là cite bắt buộc.[^1]

## Chỗ cần chỉnh trong mô tả của bạn

Mô tả hiện tại rất tốt để làm documentation nội bộ, nhưng khi viết vào thesis/paper thì nên chỉnh vài điểm:

- **CombiGCN**: paper gốc nhấn mạnh **user-user weighted graph + user-item graph**, không phải item-item similarity graph.[^5][^1]
- **BM3**: đúng là có bootstrap contrastive learning, nhưng bạn nên cẩn thận nếu ghi “image/text projector” theo kiểu implementation riêng của bạn, vì paper gốc mô tả ở mức multimodal latent representations chứ không nhất thiết đúng đúng như pipeline code của bạn.[^2]
- **FREEDOM**: rất phù hợp với mô tả “freeze item-item structure + denoise user-item graph”.[^4]


## Gợi ý cách viết cite trong luận văn

Bạn có thể viết kiểu này:

- “CombiGCN extends LightGCN by propagating embeddings on both the user-item graph and a user-user weighted graph.”[^1]
- “BM3 learns multi-modal user and item representations through bootstrap latent contrastive views without requiring auxiliary graphs or negative sampling.”[^2]
- “FREEDOM freezes the item-item graph and denoises the user-item interaction graph simultaneously for multimodal recommendation.”[^4]

Nếu muốn, mình có thể làm tiếp cho bạn một bảng **“Model → Paper → URL → Mô tả đúng chuẩn paper”** để bạn copy thẳng vào Chapter 3 hoặc Appendix.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/abs/2503.21471

[^2]: https://arxiv.org/abs/2207.05969

[^3]: https://dl.acm.org/doi/10.1145/3581783.3611943

[^4]: https://arxiv.org/abs/2211.06924

[^5]: https://dl.acm.org/doi/10.1007/978-981-97-0669-3_11

[^6]: https://arxiv.org/pdf/2503.21471.pdf

[^7]: https://www.themoonlight.io/en/review/combigcn-an-effective-gcn-model-for-recommender-system

[^8]: https://developer.nvidia.com/blog/archigan-generative-stack-apartment-building-design/

[^9]: https://arxiv.org/abs/2303.09833

[^10]: https://www.bmc.com/blogs/machine-learning-architecture/



[^15]: https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_FreeDoM_Training-Free_Energy-Guided_Conditional_Diffusion_Model_ICCV_2023_paper.pdf

[^16]: https://arxiv.org/html/2503.21471v1

[^17]: https://www.gearpatrol.com/cars/a556316/bmw-models/

[^18]: https://arxiv.org/pdf/2211.06924.pdf

[^19]: https://github.com/enoche/FREEDOM

[^20]: https://github.com/Qrange-group/Mirror-Gradient

[^21]: https://ieeexplore.ieee.org/iel8/69/4358933/11420250.pdf

[^22]: http://mmsports.multimedia-computing.de/mmsports2023/cfp.html

[^23]: https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems

[^24]: https://www.semanticscholar.org/paper/A-Tale-of-Two-Graphs:-Freezing-and-Denoising-Graph-Zhou-Shen/d0e0552e97407af6371cb82b1f0ffd853fad4223

[^25]: https://www.acmmm2023.org/reviewer-and-area-chair-guidelines/

[^26]: https://www.arxiv.org/pdf/2505.04960.pdf

[^27]: https://www.atailab.cn/seminar2025Spring/pdf/2025_AAAI_Seeing%20Beyond%20Noise%20Joint%20Graph%20Structure%20Evaluation%20and%20Denoising.pdf

[^28]: https://2026.acmmm.org/site/review-process-guidelines.html

[^29]: https://github.com/hongyurain/Recommendation-with-modality-information

[^30]: https://sxkdz.github.io/files/publications/ACMMM/LATTICE/LATTICE.pdf

