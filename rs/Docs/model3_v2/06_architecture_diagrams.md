# 06 — Architecture Diagrams: 3 Models x 4 sim_types

> Dua tren Fig.1 cua paper CombiGCN, ve lai cho ca 3 models.
> Moi model co 4 variants theo sim_type.

---

## 1. CombiGCN Architecture

```
Paper goc: CombiGCN = LightGCN + Item Similarity Graph

CombiGCN KHONG dung raw embeddings.
Multimodal features da duoc xu ly NGOAI model (Data layer)
thanh similarity matrix truoc khi dua vao.
```

### CombiGCN — sim_type = img_only

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        PREDICTION LAYER                                ║
║                                                                        ║
║                    ŷ = e*_u ⊤ · e*_i                                   ║
║                     ↑           ↑                                      ║
║                   E*_U  ⊕     E*_I  ⊕      ← mean(layer 0..K)         ║
╠══════════════════════════════════════════════════════════════════════════╣
║                     PROPAGATION LAYERS (4 layers)                      ║
║                                                                        ║
║  ┌─────────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ║
║  │  User branch (CF)   │  │  User branch     │  │  Item branch     │  ║
║  │                     │  │  (similarity)     │  │  (CF + Sim)      │  ║
║  │  E^l_U_R = R̃·E^l-1 │  │                  │  │                  │  ║
║  │                     │  │  (khong co cho    │  │  E^l_I = R^T·E^l │  ║
║  │  Layer 4: E^4_U_R   │  │   user branch)   │  │        + W·E^l-1 │  ║
║  │  Layer 3: E^3_U_R   │  │                  │  │                  │  ║
║  │  Layer 2: E^2_U_R   │  │                  │  │  Layer 4: E^4_I  │  ║
║  │  Layer 1: E^1_U_R   │  │                  │  │  Layer 3: E^3_I  │  ║
║  │       ↑              │  │                  │  │  Layer 2: E^2_I  │  ║
║  │       │              │  │                  │  │  Layer 1: E^1_I  │  ║
║  │  interaction_adj     │  │                  │  │       ↑      ↑   │  ║
║  │  (user-item graph)   │  │                  │  │       │      │   │  ║
║  └───────┬──────────────┘  └──────────────────┘  │  inter_adj  W│   │  ║
║          │                                       └───┬──────────┘   ║  ║
║          │                                           │              ║  ║
╠══════════╪═══════════════════════════════════════════╪══════════════╬══╣
║          │           EMBEDDING LAYER                 │              ║  ║
║          │                                           │              ║  ║
║        E^0_U                                       E^0_I           ║  ║
║     [553, 512]                                  [2194, 512]        ║  ║
║     Xavier init                                 Xavier init        ║  ║
║                                                                    ║  ║
╠════════════════════════════════════════════════════════════════════════╣
║                    DATA LAYER (NGOAI MODEL)                          ║
║                                                                      ║
║  interaction_adj:  D^-0.5 · A_bipartite · D^-0.5   [2747, 2747]     ║
║                                                                      ║
║  W (similarity):   cosine_sim(image_embeddings)     [2194, 2194]     ║
║                    → threshold 0.5 → normalize                       ║
║                    → cache .npz                                      ║
║                                                                      ║
║  Source: items_features.csv → feature3 (image vector string)         ║
╚══════════════════════════════════════════════════════════════════════╝
```

### CombiGCN — sim_type = tfidf

```
Giong img_only, CHI KHAC o Data layer:

  W (similarity):   TF-IDF(feature1 + feature2)        [2194, 2194]
                    → cosine_sim → threshold 0.5
                    
  Source: items_features.csv → feature1 + feature2 (text)
```

### CombiGCN — sim_type = multimodal

```
Giong img_only, CHI KHAC o Data layer:

  W (similarity):   alpha * text_sim + (1-alpha) * img_sim    [2194, 2194]
                    → threshold 0.5 → normalize

  text_sim:  BERT(feature1+feature2) → cosine_sim
  img_sim:   feature3 (image vectors) → cosine_sim
  alpha = 0.5  (late fusion)
```

### CombiGCN — sim_type = none (LightGCN thuan)

```
╔══════════════════════════════════════════════════════╗
║                  PREDICTION LAYER                    ║
║              ŷ = e*_u ⊤ · e*_i                      ║
║               ↑           ↑                          ║
║             E*_U  ⊕     E*_I  ⊕                     ║
╠══════════════════════════════════════════════════════╣
║            PROPAGATION LAYERS (4 layers)             ║
║                                                      ║
║    ego_emb = concat(user_emb, item_emb)              ║
║                                                      ║
║    Layer 1: ego = interaction_adj @ ego               ║
║    Layer 2: ego = interaction_adj @ ego               ║
║    Layer 3: ego = interaction_adj @ ego               ║
║    Layer 4: ego = interaction_adj @ ego               ║
║                                                      ║
║    KHONG CO similarity branch                        ║
║                                                      ║
╠══════════════════════════════════════════════════════╣
║             EMBEDDING LAYER                          ║
║    E^0_U [553, 512]    E^0_I [2194, 512]             ║
╠══════════════════════════════════════════════════════╣
║             DATA LAYER                               ║
║    interaction_adj [2747, 2747]    (CHI CO 1 GRAPH)  ║
║    W = None                                          ║
╚══════════════════════════════════════════════════════╝
```

### CombiGCN — Tom tat su khac biet

```
                   img_only         tfidf           multimodal        none
                   ────────         ─────           ──────────        ────
Model:             GIONG NHAU — chi dung interaction_adj + W (neu co)
                   
Data layer (W):    cosine(img)      cosine(tfidf)   a·txt+(1-a)·img   None
                   ↑                ↑                ↑                 ↑
                   feature3         feature1+2       BERT+feature3     —
```

---

## 2. BM3 Architecture

```
BM3 = LightGCN (CF) + Modal Projectors + Bootstrap Contrastive
Khong co item-item graph. Multimodal xu ly TRONG model.
```

### BM3 — sim_type = multimodal (day du nhat)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                          PREDICTION LAYER                               ║
║                                                                         ║
║                      ŷ = e*_u ⊤ · e*_i                                  ║
║                       ↑           ↑                                      ║
║                     E*_U        E*_I = item_cf ⊕ modal_emb              ║
║                       │           │         │                            ║
║                       │           │    ┌────┘                            ║
║                       │           │    │                                 ║
╠═══════════════════════╪═══════════╪════╪═════════════════════════════════╣
║  LOSS COMPUTATION     │           │    │                                 ║
║                       │           │    │                                 ║
║  L = L_bpr + L_reg + 0.2 * L_bootstrap                                  ║
║       │                    │                                             ║
║       │         ┌──────────┴──────────────────────────┐                  ║
║       │         │  Bootstrap CL (khong can negatives)  │                  ║
║       │         │                                      │                  ║
║       │         │  ┌──────┐    ┌──────┐    ┌────────┐ │                  ║
║       │         │  │item  │    │modal │    │item    │ │                  ║
║       │         │  │_cf   │←──→│_emb  │←──→│_target │ │                  ║
║       │         │  └──────┘    └──────┘    └────────┘ │                  ║
║       │         │   online      bridge      EMA target │                  ║
║       │         │                                      │                  ║
║       │         │  loss = 2 - 2·cos(pred(a), b.detach) │                  ║
║       │         └──────────────────────────────────────┘                  ║
║       │                                                                  ║
║  BPR: softplus(-(pos_score - neg_score))                                 ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                      PROPAGATION LAYERS                                  ║
║                                                                          ║
║  ┌──── CF Branch (LightGCN) ────┐    ┌──── Modal Branch ──────────────┐ ║
║  │                               │    │                                │ ║
║  │  ego = cat(user, item)        │    │  image_feats ──→ Linear(512)  │ ║
║  │                               │    │                    → img      │ ║
║  │  Layer 1: ego = adj @ ego     │    │                               │ ║
║  │  Layer 2: ego = adj @ ego     │    │  text_feats  ──→ Linear(512)  │ ║
║  │  Layer 3: ego = adj @ ego     │    │                    → txt      │ ║
║  │  Layer 4: ego = adj @ ego     │    │                               │ ║
║  │                               │    │  modal_emb = (img + txt) / 2  │ ║
║  │  final = mean(layer 0..4)     │    │              ↑                │ ║
║  │  user_cf = final[:553]        │    │         LATE FUSION            │ ║
║  │  item_cf = final[553:]        │    │                               │ ║
║  └───────────────────────────────┘    └───────────────────────────────┘ ║
║          ↑                                      ↑                       ║
║     interaction_adj                        raw embeddings               ║
║     [2747, 2747]                                                        ║
║                                                                          ║
║  ┌──── EMA Target Branch ────────┐                                      ║
║  │  item_emb_target (frozen)     │                                      ║
║  │  = 0.995 * target + 0.005 *   │                                      ║
║  │    online (moi step)          │                                      ║
║  │  → propagate → item_target    │                                      ║
║  └───────────────────────────────┘                                      ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                        EMBEDDING LAYER                                   ║
║                                                                          ║
║  E^0_U [553, 512]        E^0_I [2194, 512]       E^0_I_target (EMA)     ║
║  Xavier init              Xavier init              copy of E^0_I         ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                          INPUT LAYER                                     ║
║                                                                          ║
║  interaction_adj [2747, 2747]   (bipartite graph — giong CombiGCN)       ║
║                                                                          ║
║  image_embeddings.npy [2194, 512]  ← CLIP hoac MobileNetV2              ║
║  text_embeddings.npy  [2194, 768]  ← TF-IDF vectors                     ║
║                                                                          ║
║  ★ KHONG CO similarity matrix — dung raw embeddings truc tiep            ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### BM3 — 4 sim_types: CHI KHAC o Modal Branch

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODAL BRANCH — 4 VARIANTS                            │
│                                                                         │
│  ┌─── img_only ──────┐  ┌─── tfidf ─────────┐                         │
│  │                    │  │                    │                         │
│  │  image [2194,512]  │  │  text [2194,768]   │                         │
│  │      │             │  │      │             │                         │
│  │  Linear(512→512)   │  │  Linear(768→512)   │                         │
│  │      │             │  │      │             │                         │
│  │  modal_emb         │  │  modal_emb         │                         │
│  │  [2194, 512]       │  │  [2194, 512]       │                         │
│  │                    │  │                    │                         │
│  │  text_feats = None │  │  image_feats= None │                         │
│  └────────────────────┘  └────────────────────┘                         │
│                                                                         │
│  ┌─── multimodal ─────────────┐  ┌─── multimodal_attention ──────────┐ │
│  │                             │  │                                    │ │
│  │  image [2194,512]           │  │  image [2194,512]                  │ │
│  │      │                      │  │      │                             │ │
│  │  Linear(512→512) → img      │  │  Linear(512→512) → img            │ │
│  │                             │  │                                    │ │
│  │  text  [2194,768]           │  │  text  [2194,768]                  │ │
│  │      │                      │  │      │                             │ │
│  │  Linear(768→512) → txt      │  │  Linear(768→512) → txt            │ │
│  │                             │  │                                    │ │
│  │  modal = (img + txt) / 2    │  │  concat = cat(img,txt) [2194,1024]│ │
│  │          ↑                  │  │      │                             │ │
│  │     TRUNG BINH CONG         │  │  Linear(1024→512)                  │ │
│  │                             │  │      │                             │ │
│  │  modal_emb [2194, 512]      │  │  modal_emb [2194, 512]            │ │
│  │                             │  │      ↑                             │ │
│  │                             │  │  HOC WEIGHT TU DONG                │ │
│  └─────────────────────────────┘  └────────────────────────────────────┘ │
│                                                                         │
│  Sau buoc nay → modal_emb [2194, 512] cho moi variant                   │
│  → Cong voi item_cf → item_final                                        │
│  → Phia sau (BPR, Bootstrap CL, EMA) GIONG NHAU                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. FREEDOM Architecture

```
FREEDOM = LightGCN (CF) + Frozen kNN Graph + Content Propagation + InfoNCE
Co 2 luong propagation doc lap.
```

### FREEDOM — sim_type = multimodal (day du nhat)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                          PREDICTION LAYER                               ║
║                                                                         ║
║                      ŷ = e*_u ⊤ · e*_i                                  ║
║                       ↑           ↑                                      ║
║                     E*_U        E*_I = item_cf ⊕ item_content            ║
║                                         │              │                 ║
║                                    (from CF)     (from kNN)              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  LOSS COMPUTATION                                                       ║
║                                                                          ║
║  L = L_bpr + L_reg + 0.1 * L_infonce                                    ║
║       │                     │                                            ║
║       │         ┌───────────┴────────────────────────────┐               ║
║       │         │  InfoNCE CL (CAN negative samples)      │               ║
║       │         │                                          │               ║
║       │         │  z1 = normalize(item_cf[pos])            │               ║
║       │         │  z2 = normalize(item_content[pos])       │               ║
║       │         │                                          │               ║
║       │         │  logits = (z1 @ z2.T) / 0.2              │               ║
║       │         │           ┌─────────────────┐            │               ║
║       │         │           │ + + - - - - - - │ ← batch    │               ║
║       │         │           │ - + + - - - - - │   x batch  │               ║
║       │         │           │ - - + + - - - - │            │               ║
║       │         │           │    diagonal = ⊕  │            │               ║
║       │         │           │    off-diag  = ⊖  │            │               ║
║       │         │           └─────────────────┘            │               ║
║       │         │  loss = CrossEntropy(logits, labels)      │               ║
║       │         └──────────────────────────────────────────┘               ║
║       │                                                                    ║
║  BPR: softplus(-(pos_score - neg_score))                                   ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                      PROPAGATION LAYERS (2 branches)                      ║
║                                                                            ║
║  ┌──── CF Branch ────────────────┐  ┌──── Content Branch ──────────────┐  ║
║  │   (GIONG BM3/CombiGCN)        │  │   (FREEDOM RIENG)                │  ║
║  │                                │  │                                  │  ║
║  │  ego = cat(user, item)         │  │  modal_emb = (proj_img+proj_txt) │  ║
║  │                                │  │              / 2                 │  ║
║  │  L1: ego = inter_adj @ ego     │  │  [2194, 512]                    │  ║
║  │  L2: ego = inter_adj @ ego     │  │       │                         │  ║
║  │  L3: ego = inter_adj @ ego     │  │  L1: emb = kNN_graph @ emb      │  ║
║  │  L4: ego = inter_adj @ ego     │  │  L2: emb = kNN_graph @ emb      │  ║
║  │                                │  │  L3: emb = kNN_graph @ emb      │  ║
║  │  user_cf = mean(layers)[:553]  │  │  L4: emb = kNN_graph @ emb      │  ║
║  │  item_cf = mean(layers)[553:]  │  │                                  │  ║
║  │                                │  │  item_content = mean(layers)     │  ║
║  │  ↑                             │  │  [2194, 512]                    │  ║
║  │  interaction_adj               │  │       ↑                         │  ║
║  │  [2747, 2747]                  │  │  kNN_graph [2194, 2194]         │  ║
║  │                                │  │  (FROZEN — khong gradient)       │  ║
║  └────────────────────────────────┘  └──────────────────────────────────┘  ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                        EMBEDDING LAYER                                     ║
║                                                                            ║
║  E^0_U [553, 512]              E^0_I [2194, 512]                           ║
║  Xavier init                    Xavier init                                ║
║                                                                            ║
║  (KHONG co EMA target — khac BM3)                                          ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                    INPUT LAYER + kNN GRAPH BUILD                           ║
║                                                                            ║
║  interaction_adj [2747, 2747]                                              ║
║                                                                            ║
║  image_embeddings.npy [2194, 512]                                          ║
║  text_embeddings.npy  [2194, 768]                                          ║
║                                                                            ║
║  ★ kNN Graph (built at __init__, then FROZEN):                             ║
║                                                                            ║
║    img = normalize(image_feats)          [2194, 512]                       ║
║    txt = normalize(text_feats)           [2194, 768]                       ║
║    feats = normalize(concat(img, txt))   [2194, 1280]                      ║
║         │                                                                  ║
║    cosine_sim (batch 256) → top-10 neighbors                               ║
║         │                                                                  ║
║    symmetrize (A + A^T) → row-normalize                                    ║
║         │                                                                  ║
║    kNN_graph [2194, 2194] sparse → register_buffer (NO GRADIENT)           ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### FREEDOM — 4 sim_types: KHAC o 2 CHO

```
┌─────────────────────────────────────────────────────────────────────────┐
│              ★ CHO 1: kNN GRAPH BUILD (tai init, 1 lan)                 │
│                                                                         │
│  img_only:      feats = norm(image)             [2194, 512]             │
│  tfidf:         feats = norm(text)              [2194, 768]             │
│  multimodal:    feats = norm(cat(img, txt))      [2194, 1280]           │
│  mm_attention:  feats = norm(cat(img, txt))      [2194, 1280]  ← GIONG │
│                                                                         │
│  → cosine_sim → top-10 → symmetrize → normalize → FREEZE               │
│                                                                         │
│  ⚠ multimodal va mm_attention co CUNG kNN graph!                       │
│    Chung chi khac nhau o modal projection (cho 2 ben duoi)              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│              ★ CHO 2: MODAL PROJECTION (moi forward)                    │
│                                                                         │
│  img_only:      modal = proj_img(image_feats)                           │
│  tfidf:         modal = proj_txt(text_feats)                            │
│  multimodal:    modal = (proj_img + proj_txt) / 2                       │
│  mm_attention:  modal = Linear(cat(proj_img, proj_txt))                 │
│                                                                         │
│  → modal_emb [2194, 512]                                                │
│  → Content propagation: kNN_graph @ modal_emb (4 layers)                │
│  → item_content [2194, 512]                                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│              TU DAY TRO DI: GIONG NHAU CA 4 VARIANTS                    │
│                                                                         │
│  item_final = item_cf + item_content                                    │
│  BPR loss + L2 reg + InfoNCE                                            │
│  backward → update (kNN graph KHONG update)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. So sanh tong the — 3 Models cung 1 hinh

```
                    CombiGCN              BM3                  FREEDOM
                    ════════              ═══                  ═══════

Prediction:         u · i                 u · i                u · i
                     ↑                     ↑                    ↑

Fusion:         item_cf + item_sim    item_cf + modal      item_cf + item_content
                (moi layer)           (cuoi cung)          (cuoi cung)
                     ↑       ↑             ↑       ↑            ↑          ↑
                     │       │             │       │            │          │

Propagation:    ┌────┴──┐ ┌──┴───┐    ┌────┴──┐ ┌──┴───┐  ┌────┴──┐ ┌────┴─────┐
                │LightGCN│ │sim@  │    │LightGCN│ │Linear│  │LightGCN│ │kNN@modal │
                │(inter  │ │item  │    │(inter  │ │proj  │  │(inter  │ │(content  │
                │ adj)   │ │_emb  │    │ adj)   │ │      │  │ adj)   │ │ propag.) │
                └────────┘ └──────┘    └────────┘ └──────┘  └────────┘ └──────────┘
                     ↑       ↑             ↑       ↑            ↑          ↑
                     │       │             │       │            │          │
Input:          inter_adj  sim_adj     inter_adj  raw .npy  inter_adj  raw .npy
                           (precomp)              tensors              tensors
                                                                      + kNN graph
                                                                      (frozen)

CL Loss:            KHONG              Bootstrap            InfoNCE
                                       (no negatives        (co negatives
                                        EMA target)          tu batch)

Extra:              —                  EMA encoder          Frozen kNN
                                       + predictor          graph

sim_type            Data layer         Model layer          Model layer
affect:             (matrix W)         (projection)         (projection
                                                            + kNN build)
```
