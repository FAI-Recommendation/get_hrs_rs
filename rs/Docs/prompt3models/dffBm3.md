# BM3 — Adapted Architecture for Fashion Recommendation

> Paper goc: "Bootstrap Latent Representations for Multi-modal Recommendation" — Zhou et al., WWW 2023.
> Implementation nay duoc dieu chinh (adapted) cho bai toan fashion recommendation voi dataset 553 users, 2194 items.

---

## 1. Tong quan kien truc

BM3 hoc bieu dien items tu **2 nguon thong tin**:

- **CF Branch** (nhanh trai — mau xanh): Hoc tu hanh vi tuong tac user-item bang GCN propagation.
- **Modal Branch** (nhanh phai — mau cam): Hoc tu dac trung noi dung (image/text) bang Linear Projectors — **KHONG qua graph** nao.

Hai nhanh duoc **align** (dong bo) bang **Bootstrap Contrastive Learning** — co che tu hoc KHONG can negative samples, chi dung EMA target encoder + predictor head.

---

## 2. Input Layer (duoi cung)

### 2.1 ID Embeddings (duoi trai)

Moi user va item deu co 1 vector embedding rieng:

$$h_i^0 \in \mathbb{R}^{2194 \times 512}, \quad h_u^0 \in \mathbb{R}^{553 \times 512}$$

- Khoi tao: Xavier Normal
- Day la cac tham so **hoc duoc** (learnable).

Ngoai ra, BM3 con co **EMA target copy**:

$$h_{i\_target}^0 \in \mathbb{R}^{2194 \times 512}$$

- La ban sao cua $h_i^0$, nhung **dong bang** (frozen) — chi duoc cap nhat qua EMA (xem muc 5).
- Khong nhan gradient tu optimizer.

### 2.2 Raw Features + Linear Projectors (duoi phai)

Dac trung noi dung cua item duoc trich xuat truoc (pre-extracted):

| Feature | Kich thuoc | Nguon |
|---|---|---|
| `image_feats` | [2194 × 512] | CLIP hoac MobileNetV2 |
| `text_feats` | [2194 × 768] | TF-IDF vectors |

Cac features duoc chieu xuong khong gian chung **512 chieu** bang **Linear Projectors** (1 layer tuyen tinh, khong co activation function):

$$\text{proj\_img}: \mathbb{R}^{512} \rightarrow \mathbb{R}^{512}, \quad \text{proj\_txt}: \mathbb{R}^{768} \rightarrow \mathbb{R}^{512}$$

Tuy vao **sim_type**, cach ket hop khac nhau:

| sim_type | Cach tinh `modal_emb` |
|---|---|
| `img_only` | `proj_img(image_feats)` |
| `tfidf` | `proj_txt(text_feats)` |
| `multimodal` | `(proj_img + proj_txt) / 2` — trung binh (late fusion) |
| `mm_attention` | `Linear(concat(proj_img, proj_txt))` — hoc weight tu dong |

Ket qua: **1 vector `modal_emb` [2194 × 512]** duy nhat.

> **Diem quan trong:** modal_emb di **thang** vao fusion — KHONG qua bat ky graph propagation nao. Day la diem khac biet co ban voi FREEDOM (truyen qua kNN graph).

---

## 3. CF Branch — Nhanh Collaborative Filtering (trai, xanh duong)

Nhanh nay hoc tu **hanh vi tuong tac** (ai mua gi) bang GCN propagation.

### Buoc 1: Noi embeddings

$$\text{ego} = \text{concat}(h_i^0, h_u^0) \in \mathbb{R}^{2747 \times 512}$$

### Buoc 2: GCN Propagation (4 layers)

Moi layer: nhan ma tran ke voi embedding hien tai:

$$\text{ego}^{(l)} = A_{\text{interaction}} \times \text{ego}^{(l-1)}$$

Trong do $A_{\text{interaction}} \in \mathbb{R}^{2747 \times 2747}$ la ma tran ke bipartite da normalize ($D^{-1/2}AD^{-1/2}$).

Khi training, ap dung **random edge dropout** de regularize.

### Buoc 3: Mean pooling

$$\text{final} = \text{mean}(\text{ego}^{(0)}, \text{ego}^{(1)}, \text{ego}^{(2)}, \text{ego}^{(3)}, \text{ego}^{(4)})$$

### Buoc 4: Tach ra user va item

$$\text{user\_cf} \in \mathbb{R}^{553 \times 512}, \quad \text{item\_cf} \in \mathbb{R}^{2194 \times 512}$$

> **Luu y:** GCN propagation chay **2 lan** moi forward pass — 1 lan voi online embeddings ($h_i^0$), 1 lan voi target embeddings ($h_{i\_target}^0$) de tao ra `item_target` cho bootstrap loss.

---

## 4. Fusion — Ket hop 2 nhanh

$$h_i = \text{item\_cf} + \text{modal\_emb}$$

- `item_cf`: bieu dien item tu hanh vi mua sam (CF view)
- `modal_emb`: bieu dien item tu hinh anh/mo ta (Modal view)

---

## 5. Bootstrap Contrastive Learning (phan quan trong nhat)

Day la **diem doc dao nhat** cua BM3 — hoc alignment giua 2 views ma **KHONG can negative samples**.

### 5.1 Ba thanh phan chinh

| Thanh phan | Ky hieu | Nguon | Vai tro |
|---|---|---|---|
| **Online CF View** | `item_cf` | Tu GCN propagation (online encoder) | View hoc tu tuong tac |
| **Modal View** | `modal_emb` | Tu Linear Projectors | View hoc tu noi dung |
| **EMA Target View** | `item_target` | Tu GCN propagation (target encoder) | "Anchor" on dinh |

### 5.2 Predictor Head

Mot MLP nho **chi dung tren online branch**:

$$\text{Predictor}: \text{Linear}(512 \rightarrow 512) \rightarrow \text{ReLU} \rightarrow \text{Linear}(512 \rightarrow 512)$$

Predictor tao ra **bat doi xung** (asymmetry) giua online va target — day la yeu to then chot ngan **collapse** (hien tuong tat ca embeddings hoi tu ve cung 1 diem).

### 5.3 Bootstrap Loss

$$\mathcal{L}_{\text{boot}}(\text{online}, \text{target}) = 2 - 2 \cdot \text{mean}\left[\text{normalize}(\text{Predictor}(\text{online})) \cdot \text{normalize}(\text{target.detach()})\right]$$

- `online`: qua Predictor roi normalize
- `target`: chi normalize roi **detach** (cat gradient — khong hoc nguoc lai)
- Gia tri: 0 (hoan toan giong) den 4 (hoan toan nguoc)

Tong contrastive loss:

$$\mathcal{L}_{\text{bootstrap}} = \frac{\mathcal{L}_{\text{boot}}(\text{item\_cf}, \text{modal\_emb}) + \mathcal{L}_{\text{boot}}(\text{modal\_emb}, \text{item\_target})}{2}$$

- **Term 1** `L_boot(item_cf, modal_emb)`: Ep CF view va Modal view cua cung 1 item gan nhau
- **Term 2** `L_boot(modal_emb, item_target)`: Ep Modal view gan voi phien ban "on dinh" cua item (EMA target)

### 5.4 EMA Update

Sau **moi training step**, target encoder duoc cap nhat **cham** tu online encoder:

$$\theta_{\text{target}} = 0.995 \times \theta_{\text{target}} + 0.005 \times \theta_{\text{online}}$$

- Momentum = 0.995 → target **rat cham thay doi** → dong vai tro "anchor" on dinh
- Khong dung optimizer — chi copy trung binh dong

> **Tai sao can EMA + Predictor?** Bootstrap CL khong co negative samples → model co the "luoi" — tra ve embeddings giong nhau cho moi item (collapse). EMA target cung cap diem neo on dinh, Predictor tao bat doi xung → 2 yeu to nay phoi hop ngan collapse hieu qua (da duoc chung minh trong BYOL paper cua DeepMind).

---

## 6. Loss Layer (tren cung)

### 6.1 BPR Loss — Xep hang

$$\mathcal{L}_{\text{BPR}} = \text{mean}\left[\text{softplus}\left(-(s_{\text{pos}} - s_{\text{neg}})\right)\right]$$

Trong do:

$$s_{\text{pos}} = h_u^T \cdot h_{i\_pos}, \quad s_{\text{neg}} = h_u^T \cdot h_{i\_neg}$$

### 6.2 L2 Regularization

$$\mathcal{L}_{\text{reg}} = \lambda \cdot \frac{\|h_u^0\|^2 + \|h_{pos}^0\|^2 + \|h_{neg}^0\|^2}{\text{batch\_size}}$$

### 6.3 Tong loss

$$\mathcal{L} = \mathcal{L}_{\text{BPR}} + \mathcal{L}_{\text{reg}} + 0.2 \times \mathcal{L}_{\text{bootstrap}}$$

| Thanh phan | Vai tro |
|---|---|
| $\mathcal{L}_{\text{BPR}}$ | Dam bao chat luong ranking |
| $\mathcal{L}_{\text{reg}}$ | Ngan overfit |
| $\mathcal{L}_{\text{bootstrap}}$ | Dong bo CF view va Modal view ma khong can negatives |

---

## 7. Prediction (khi inference)

$$\hat{y} = h_u^T \cdot h_i = \text{user\_cf} \cdot (\text{item\_cf} + \text{modal\_emb})^T$$

- Khong su dung Predictor khi inference — chi dung khi training.
- Khong su dung EMA target — chi dung raw embeddings.
- Chon **top-K items** co diem cao nhat de recommend.

---

## 8. Hyperparameters

| Tham so | Gia tri | Y nghia |
|---|---|---|
| `embedding_dim` | 512 | Kich thuoc embedding |
| `n_layers` | 4 | So layers GCN |
| `lr` | 0.001 | Learning rate |
| `momentum` | 0.995 | He so EMA (gan 1 = cap nhat cham) |
| `cl_weight` | 0.2 | Trong so Bootstrap CL loss |
| `decay` | 1e-4 | He so L2 regularization |

---

## 9. Diem khac biet so voi paper goc va ly do

### 9.1 Gop modal thanh 1 `modal_emb` (paper giu rieng h_v, h_t)

**Paper:** Tinh contrastive loss **rieng** cho tung modality — `L_align(h_v, h_i)` + `L_align(h_t, h_i)` — tao ra 2|M| contrastive pairs.

**Implementation:** Gop thanh 1 `modal_emb = (proj_img + proj_txt) / 2` roi tinh 2 bootstrap loss terms.

> **Ly do:** Dataset fashion nho (2194 items) — tinh rieng per modality se tao ra **qua nhieu loss terms** (4 pairs thay vi 2), de dan den **gradient conflict** giua cac modalities. Gop lai giup **on dinh training** va giam so luong hyperparameters can tune. Ngoai ra, late fusion (trung binh) cho phep 2 modalities **bo sung** cho nhau thay vi canh tranh — phu hop voi fashion domain noi ca hinh anh va mo ta deu quan trong ngang nhau.

### 9.2 Bo L_rec va L_mask (paper co 4 loss, code co 3)

**Paper:** 4 loss terms:
- L_rec: Graph Reconstruction — align user ↔ item qua dropout augmentation
- L_align: Inter-modality — align modal ↔ item target
- L_mask: Intra-modality — self-align modal voi phien ban dropout cua no
- L_reg: Regularization

**Implementation:** 3 loss terms: L_bpr + L_bootstrap + L_reg.

> **Ly do:** L_rec (graph reconstruction) va BPR loss co **chuc nang tuong tu** — deu ep user-item pair co diem cao. Giu ca 2 se **redundant** va gay gradient conflict. BPR loss da duoc chung minh hieu qua cho ranking trong nhieu recommendation systems — nen thay the L_rec bang BPR de don gian hoa ma van giu hieu qua.
>
> L_mask (intra-modality dropout) la co che tu augmentation — tao "noise" roi tu hoc de khoi phuc. Tren dataset nho, augmentation nay **de overfit** vao noise pattern cua training set. Bo L_mask giup model **tap trung hoc signal chinh** (CF ↔ Modal alignment) thay vi bi phan tan boi qua nhieu auxiliary losses.

### 9.3 Bo dropout augmentation tren target view

**Paper:** Tao contrastive views bang `h_target = h · Bernoulli(ρ)` — dropout ngau nhien.

**Implementation:** Chi dung `target.detach()` — khong dropout.

> **Ly do:** Dropout augmentation tao ra **stochastic noise** khac nhau moi step — giup model hoc bieu dien robust tren dataset lon (Baby: 19k users, Electronics: 192k users). Tren dataset nho (553 users), noise nay **khong du da dang** de co y nghia va co the lam **mat on dinh training**. Detach (stop-gradient) da du de ngan gradient chay nguoc ve target — ket hop voi EMA update la du de ngan collapse.

### 9.4 Khong dung Predictor khi inference

**Paper:** Inference dung predictor output: $r(h_u, h_i) = \tilde{h}_u \cdot \tilde{h}_i$

**Implementation:** Inference dung raw embeddings: $r(h_u, h_i) = h_u \cdot h_i$

> **Ly do:** Predictor head duoc thiet ke de **tao asymmetry trong training** (ngan collapse), khong phai de cai thien bieu dien. Khi inference, predictor co the **distort** embeddings — dac biet khi Predictor chua hoc du tot tren dataset nho. Thuc nghiem cho thay dung raw embeddings cho ket qua **on dinh hon** — cung la cach lam cua BYOL goc (chi dung predictor khi training).

### 9.5 Fusion bang modal_emb thay vi residual H⁰

**Paper:** `H_i = READOUT(H⁰...Hᴸ) + H⁰` — cong them ID embedding ban dau (residual connection).

**Implementation:** `h_i = item_cf + modal_emb` — cong modal features.

> **Ly do:** Residual H⁰ giup **giu thong tin ID goc** khi GCN propagate nhieu layers (chong over-smoothing). Tuy nhien, voi chi 4 layers tren dataset nho, over-smoothing **chua xay ra nghiem trong**. Thay vao do, cong modal_emb mang lai loi ich lon hon — **bo sung thong tin noi dung** (hinh anh, mo ta) vao bieu dien item, dac biet huu ich cho items co it tuong tac (near cold-start). Day la trade-off: paper uu tien chong over-smoothing, implementation uu tien **content enrichment**.

### 9.6 Linear Projectors (1 layer) — giong paper

**Paper (Eq 1):** `h_m = e_m · W_m + b_m` — goi la "MLP" nhung thuc te chi 1 layer.

**Implementation:** `nn.Linear(dim_in, dim_out)` — chinh xac giong cong thuc paper.

> Day la diem **giu nguyen** tu paper — projection 1 layer da du de chuyen doi khong gian features ma khong can non-linearity.

---

## 10. Tom tat

| # | Thanh phan | Paper BM3 (WWW 2023) | Implementation | Ly do thay doi |
|---|---|---|---|---|
| 1 | Modal contrastive | Rieng per modality (h_v, h_t) | Gop thanh 1 modal_emb | On dinh training, giam gradient conflict |
| 2 | Loss terms | 4: L_rec + L_align + L_mask + L_reg | 3: L_bpr + L_bootstrap + L_reg | BPR thay L_rec (chuc nang tuong tu), bo L_mask (de overfit tren dataset nho) |
| 3 | Dropout augmentation | Co — Bernoulli(ρ) | Khong — chi detach | Noise khong du da dang tren dataset nho |
| 4 | Predictor o inference | Co | Khong | Predictor chi ngan collapse, khong cai thien bieu dien |
| 5 | Item residual | READOUT + H⁰ | item_cf + modal_emb | Uu tien content enrichment hon chong over-smoothing |
| 6 | Linear Projectors | 1-layer (goi la MLP) | 1-layer nn.Linear | **Giu nguyen** — giong paper |
