# FREEDOM — Adapted Architecture for Fashion Recommendation

> Paper goc: "A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation" — Zhou et al., ACM MM 2023.
> Implementation nay duoc dieu chinh (adapted) cho bai toan fashion recommendation voi dataset 553 users, 2194 items.

---

## 1. Tong quan kien truc

FREEDOM chia qua trinh hoc bieu dien item thanh **2 nhanh doc lap**, sau do **ket hop** lai:

- **CF Branch** (nhanh trai — mau xanh): Hoc tu hanh vi tuong tac user-item bang LightGCN.
- **Content Branch** (nhanh phai — mau xanh la): Hoc tu dac trung noi dung (image/text) bang cach truyen qua **frozen kNN item-item graph**.

Hai nhanh tao ra 2 "goc nhin" (views) khac nhau ve cung 1 item, sau do duoc **align** bang InfoNCE contrastive loss.

---

## 2. Input Layer (duoi cung)

### 2.1 ID Embeddings (duoi trai)

Moi user va item deu co 1 vector embedding rieng, khoi tao ngau nhien:

$$h_i^0 \in \mathbb{R}^{2194 \times 512}, \quad h_u^0 \in \mathbb{R}^{553 \times 512}$$

- Khoi tao: Xavier Normal
- Day la cac tham so **hoc duoc** (learnable) — duoc cap nhat moi training step.

### 2.2 Raw Features + Linear Projectors (duoi phai)

Dac trung noi dung cua item duoc trich xuat truoc (pre-extracted):

| Feature | Kich thuoc | Nguon |
|---|---|---|
| `image_feats` | [2194 × 512] | CLIP hoac MobileNetV2 |
| `text_feats` | [2194 × 768] | TF-IDF vectors |

Cac features nay duoc chieu (project) xuong khong gian chung **512 chieu** bang **Linear Projectors** (1 layer tuyen tinh, khong co activation function):

$$\text{proj\_img}: \mathbb{R}^{512} \rightarrow \mathbb{R}^{512}, \quad \text{proj\_txt}: \mathbb{R}^{768} \rightarrow \mathbb{R}^{512}$$

Tuy vao **sim_type**, cach ket hop khac nhau:

| sim_type | Cach tinh `modal_emb` |
|---|---|
| `img_only` | `proj_img(image_feats)` |
| `tfidf` | `proj_txt(text_feats)` |
| `multimodal` | `(proj_img + proj_txt) / 2` — trung binh (late fusion) |
| `mm_attention` | `Linear(concat(proj_img, proj_txt))` — hoc weight tu dong |

Ket qua: **1 vector `modal_emb` [2194 × 512]** duy nhat cho tat ca items.

---

## 3. CF Branch — Nhanh Collaborative Filtering (trai, xanh duong)

Nhanh nay hoc tu **hanh vi tuong tac** (ai mua gi) — giong hệt LightGCN.

### Buoc 1: Noi embeddings

$$\text{ego} = \text{concat}(h_i^0, h_u^0) \in \mathbb{R}^{2747 \times 512}$$

### Buoc 2: Truyen tin qua User-Item Graph (4 layers)

Moi layer: nhan ma tran ke voi embedding hien tai:

$$\text{ego}^{(l)} = A_{\text{interaction}} \times \text{ego}^{(l-1)}$$

Trong do $A_{\text{interaction}} \in \mathbb{R}^{2747 \times 2747}$ la ma tran ke bipartite da normalize ($D^{-1/2}AD^{-1/2}$).

Khi training, ap dung **random edge dropout** de regularize (loai ngau nhien 1 so canh).

### Buoc 3: Mean pooling

$$\text{final} = \text{mean}(\text{ego}^{(0)}, \text{ego}^{(1)}, \text{ego}^{(2)}, \text{ego}^{(3)}, \text{ego}^{(4)})$$

### Buoc 4: Tach ra user va item

$$\text{user\_cf} \in \mathbb{R}^{553 \times 512}, \quad \text{item\_cf} \in \mathbb{R}^{2194 \times 512}$$

---

## 4. Content Branch — Nhanh Noi Dung (phai, xanh la)

Day la **diem doc dao nhat** cua FREEDOM — truyen dac trung noi dung qua item-item graph.

### Step 1: Xay dung Frozen kNN Graph (❄)

Truoc khi training, FREEDOM xay dung 1 do thi item-item dua tren **tuong dong noi dung**:

1. **Chon features** theo sim_type (giong muc 2.2 — dung cung features de build graph)
2. **Tinh cosine similarity** giua tat ca cap items (xu ly theo batch 256 de tranh OOM)
3. **Chon k=10 neighbors** — moi item chi giu 10 items giong no nhat
4. **Doi xung hoa** — them canh nguoc: $S = S + S^T$
5. **Row-normalize** — chuan hoa theo hang: $S_{ij} = S_{ij} / \sum_j S_{ij}$
6. **Dong bang (Freeze)** — luu bang `register_buffer`, **KHONG co gradient, KHONG cap nhat khi training**

> **Tai sao dong bang?** Graph structure nen on dinh — cap nhat lien tuc bang gradient se dan den noisy updates va overfitting. Dong bang giup "denoise" — giu cau truc graph sach.

Ket qua: `kNN_graph [2194 × 2194]` sparse, frozen.

### Step 2: Content Propagation (4 GCN layers tren frozen graph)

Dau vao la `modal_emb` (tu Linear Projectors), **KHONG phai ID embeddings**:

$$\text{emb}^{(0)} = \text{modal\_emb}$$
$$\text{emb}^{(l)} = S_{\text{kNN}} \times \text{emb}^{(l-1)}, \quad l = 1, 2, 3, 4$$

Moi layer: moi item **"hap thu"** dac trung tu 10 neighbors gan nhat. Qua 4 layers: thong tin lan truyen xa 4 buoc (neighbor cua neighbor cua neighbor...).

Mean pooling tat ca layers:

$$\text{item\_content} = \text{mean}(\text{emb}^{(0)}, \text{emb}^{(1)}, ..., \text{emb}^{(4)}) \in \mathbb{R}^{2194 \times 512}$$

---

## 5. Fusion — Ket hop 2 nhanh

Embeddings tu 2 nhanh duoc **cong** lai:

$$h_i = \text{item\_cf} + \text{item\_content}$$

- `item_cf`: hoc tu hanh vi mua sam (CF view)
- `item_content`: hoc tu hinh anh/mo ta san pham (Content view)

`user_cf` ($h_u$) duoc giu nguyen, khong can fusion vi chi co 1 nguon (user-item graph).

---

## 6. Loss Layer (tren cung)

### 6.1 BPR Loss — Xep hang

Ep diem cua item user **da mua** cao hon item user **chua mua**:

$$\mathcal{L}_{\text{BPR}} = \text{mean}\left[\text{softplus}\left(-(s_{\text{pos}} - s_{\text{neg}})\right)\right]$$

Trong do:

$$s_{\text{pos}} = h_u^T \cdot h_{i\_pos}, \quad s_{\text{neg}} = h_u^T \cdot h_{i\_neg}$$

### 6.2 L2 Regularization

Ngan overfit bang cach phat embeddings qua lon:

$$\mathcal{L}_{\text{reg}} = \lambda \cdot \frac{\|h_u^0\|^2 + \|h_{pos}^0\|^2 + \|h_{neg}^0\|^2}{\text{batch\_size}}$$

### 6.3 InfoNCE Contrastive Loss — Align 2 views

Day la co che **ep 2 nhanh nhin giong nhau** ve cung 1 item:

$$z_1 = \text{normalize}(\text{item\_cf}[\text{pos\_items}])$$
$$z_2 = \text{normalize}(\text{item\_content}[\text{pos\_items}])$$

Tinh similarity matrix:

$$\text{logits} = \frac{z_1 \cdot z_2^T}{\tau}, \quad \tau = 0.2$$

$$\mathcal{L}_{\text{InfoNCE}} = \text{CrossEntropy}(\text{logits}, [0, 1, 2, ..., \text{batch}-1])$$

Y nghia: **duong cheo** cua similarity matrix la positive pairs (cung 1 item, 2 views khac nhau — push gan nhau), **ngoai duong cheo** la negative pairs (khac item — push xa nhau).

### 6.4 Tong loss

$$\mathcal{L} = \mathcal{L}_{\text{BPR}} + \mathcal{L}_{\text{reg}} + 0.1 \times \mathcal{L}_{\text{InfoNCE}}$$

| Thanh phan | Vai tro |
|---|---|
| $\mathcal{L}_{\text{BPR}}$ | Dam bao chat luong ranking (item dung xep tren item sai) |
| $\mathcal{L}_{\text{reg}}$ | Ngan overfit |
| $\mathcal{L}_{\text{InfoNCE}}$ | Dong bo 2 goc nhin CF va Content |

---

## 7. Prediction (khi inference)

$$\hat{y} = h_u^T \cdot h_i = \text{user\_cf} \cdot (\text{item\_cf} + \text{item\_content})^T$$

Diem cao = user co kha nang thich item do. Chon **top-K items** co diem cao nhat de recommend.

---

## 8. Hyperparameters

| Tham so | Gia tri | Y nghia |
|---|---|---|
| `embedding_dim` | 512 | Kich thuoc embedding |
| `n_layers` | 4 | So layers GCN (ca 2 nhanh) |
| `lr` | 0.001 | Learning rate |
| `knn_k` | 10 | So neighbors trong kNN graph |
| `cl_weight` | 0.1 | Trong so InfoNCE loss |
| `cl_temp` | 0.2 | Temperature cua InfoNCE |
| `decay` | 1e-4 | He so L2 regularization |

---

## 9. Diem khac biet so voi paper goc

| Thanh phan | Paper goc (ACM MM 2023) | Implementation cua chung toi |
|---|---|---|
| kNN graph | Build rieng per modality, tron bang $\alpha$ | Concat features roi build 1 graph — bo hyperparameter $\alpha$ |
| Graph type | Unweighted (0/1) + symmetric norm | Weighted (giu similarity) + row-normalize |
| Structure Denoising | Degree-sensitive edge pruning | Random dropout — dataset nho, degree phan bo deu |
| Content propagation input | ID embeddings $h_i^0$ | **modal_emb** (projected features) — semantic consistency |
| Content readout | Chi layer cuoi | **Mean all layers** — tranh over-smoothing |
| Loss | BPR + modal-specific BPR per modality | BPR + **InfoNCE** (CF view ↔ Content view) |
| Prediction | Chi dung ID embeddings | Dung ca `item_cf + item_content` — bo sung content signal |
