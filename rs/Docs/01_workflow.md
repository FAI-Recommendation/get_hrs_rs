# Workflow: CombiGCN PyG

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          INPUT FILES (dataset folder)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  train.txt          — user_id  item_id item_id ...  (purchase history)       ║
║  test.txt           — user_id  item_id item_id ...  (ground truth)           ║
║  items_features.csv — item_id, feature1(text), feature2(text), feature3(img) ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 1 — Data Loading  (load_data.py)                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Data.__init__()                                                             ║
║  ├── Đọc train.txt  →  self.R (user×item sparse), self.train_items           ║
║  ├── Đọc test.txt   →  self.test_set                                         ║
║  ├── self.U = R * R^T  (user-user co-purchase, dùng cho similar_users)       ║
║  └── Đọc items_features.csv  →  self.items_features                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║             PHASE 2 — Build Adjacency Matrices  (get_norm_adj_mat)           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  sim_type="none"        →  build index {0}                                   ║
║  sim_type="tfidf"       →  build index {0, 3}                                ║
║  sim_type="multimodal"  →  build index {0, 6}                                ║
║  sim_type="img_only"    →  build index {0, 7}                                ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  Index │ File NPZ                          │ Nguồn                  │    ║
║  │  ──────┼───────────────────────────────────┼────────────────────── │    ║
║  │    0   │ s_interaction_adj_mat.npz         │ train.txt (R matrix)   │    ║
║  │    3   │ s_tfidf_item_similarity_adj_mat   │ feature1+feature2 TF-IDF│   ║
║  │    6   │ s_multimodal_similarity_adj_mat   │ BERT text + image emb  │    ║
║  │    7   │ s_img_similarity_adj_mat          │ feature3 (image emb)   │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  Mỗi matrix → D^{-0.5} A D^{-0.5}  (symmetric normalization)                ║
║  Cache: nếu .npz đã tồn tại → load, không cần tính lại                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                        ┌───────────┴───────────┐
                        │                       │
              interaction_adj (0)      similarity_adj (3/6/7)
              shape: (U+I, U+I)        shape: (I, I)
                        │                       │
                        └───────────┬───────────┘
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 3 — Model  (model.py: CombiGCN)                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  nn.Embedding(n_users, embed_size=512)  ← khởi tạo ngẫu nhiên (Xavier)      ║
║  nn.Embedding(n_items, embed_size=512)  ← khởi tạo ngẫu nhiên (Xavier)      ║
║                                                                              ║
║  ego_emb = concat([user_emb, item_emb])   shape: (U+I, 512)                 ║
║                                                                              ║
║  ── Lặp L=4 layers ──────────────────────────────────────────────────────   ║
║  │                                                                           ║
║  │  [LightGCN thuần — sim_type="none"]                                       ║
║  │    ego_emb = interaction_adj @ ego_emb                                    ║
║  │                                                                           ║
║  │  [CombiGCN — sim_type=tfidf/multimodal/img_only]                          ║
║  │    interaction_emb  = interaction_adj @ ego_emb                           ║
║  │    user_next        = interaction_emb[:n_users]                           ║
║  │    item_interaction = interaction_emb[n_users:]                           ║
║  │    item_similar     = similarity_adj  @ item_emb      ← item-item graph   ║
║  │    item_next        = item_interaction + item_similar  ← fusion (sum)     ║
║  │    ego_emb          = concat([user_next, item_next])                      ║
║  │                                                                           ║
║  └── all_embs.append(ego_emb)                                                ║
║                                                                              ║
║  final_emb = mean(layer_0, layer_1, layer_2, layer_3, layer_4)               ║
║  user_final (U, 512),  item_final (I, 512)                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PHASE 4 — Training Loop  (train.py)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  for epoch in 1..1000:                                                       ║
║    for batch in BPR_sample():                                                ║
║      users, pos_items, neg_items = sample()                                  ║
║                                                                              ║
║      ── BPR Loss ─────────────────────────────────────────────────────────  ║
║      pos_score = u_emb · pos_emb                                             ║
║      neg_score = u_emb · neg_emb                                             ║
║      mf_loss   = mean(softplus(neg_score - pos_score))                       ║
║      reg_loss  = decay * L2(u, pos, neg) / batch_size                        ║
║      loss      = mf_loss + reg_loss                                          ║
║                                                                              ║
║      Adam.backward() → update user_emb, item_emb                            ║
║                                                                              ║
║    if epoch % eval_interval == 0:  (mỗi 40 epoch)                           ║
║      ── Evaluate ──────────────────────────────────────────────────────────  ║
║      scores = user_final @ item_final^T    shape: (batch_users, n_items)     ║
║      Mask đã mua (train items) → rank còn lại                                ║
║      Tính: Recall, Precision, NDCG, MAP, MRR, HitRatio  @K=[1,5,10,20]      ║
║                                                                              ║
║    if epoch % checkpoint_interval == 0:  (mỗi 200 epoch)                    ║
║      → save checkpoint_epoch{N}.pt                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PHASE 5 — Save & Export                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  weights/<sim_type>/                                                         ║
║  ├── best_model.pt              ← best epoch (theo Recall@1)                 ║
║  ├── best_metrics.json          ← tất cả metrics tại best epoch              ║
║  ├── checkpoint_epoch200.pt                                                  ║
║  ├── checkpoint_epoch400.pt                                                  ║
║  └── ...                                                                     ║
║                                                                              ║
║  output/<sim_type>/combigcn.result  ← best metrics dạng text                 ║
║                                                                              ║
║  tensorboard/<run_name>/            ← TensorBoard logs (train loss + test)   ║
║                                                                              ║
║  wandb (online)                     ← real-time curves                       ║
║  ├── project: WANDB_PROJECT (.env)                                           ║
║  └── run name: {sim_type}_layers4_dim512_lr0.001_reg1e-04_{clip|mbnv2}       ║
║                                                                              ║
║  HuggingFace Hub (sau khi train xong)                                        ║
║  └── HF_REPO_ID/{sim_type}/  ← push toàn bộ folder weights                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Sơ đồ phân nhánh theo sim_type

```
                         train.py
                             │
                    args.sim_type = ?
                             │
          ┌──────────────────┼──────────────────┬──────────────┐
          │                  │                  │              │
       "none"            "tfidf"         "multimodal"      "img_only"
          │                  │                  │              │
   interaction_adj    interaction_adj    interaction_adj  interaction_adj
   similarity=None    + tfidf_adj        + multimodal_adj + img_adj
          │                  │                  │              │
   LightGCN thuần      CombiGCN           CombiGCN         CombiGCN
   (chỉ user↔item)    (+ text sim)    (+ text+image sim) (+ image sim)
```

---

## Embedding flow chi tiết (1 layer CombiGCN)

```
  user_emb (U,512) ──┐                item_emb (I,512) ──────────────────┐
                      │                        │                           │
                      └──── ego_emb ───────────┘                           │
                                  │                                        │
                    interaction_adj (U+I, U+I)                             │
                          @ ego_emb                                        │
                                  │                                        │
                    ┌─────────────┴────────────┐                           │
                    │                          │                           │
             user_next (U,512)       item_interaction (I,512)              │
                    │                          │                           │
                    │                 sim_adj (I,I) @ item_emb ────────────┘
                    │                          │         item_similar (I,512)
                    │                          │
                    │                 item_next = item_interaction + item_similar
                    │                          │
                    └──── ego_emb_next ─────────┘
                         (U+I, 512) → next layer
```

---

## Scripts entry points

```
scripts/
├── test_clip/test_all.sh       → quick test 20 epoch (CLIP, verify pipeline)
├── test_mbnv2/test_all.sh      → quick test 20 epoch (MobileNetV2)
├── run_all_clip/run_all.sh     → full train 1000 epoch (CLIP, reg=1e-4)
└── run_all_mbnv2/run_all.sh    → full train 1000 epoch (MobileNetV2, reg=1e-4)
         │
         │  mỗi run_all gọi 4 scripts tuần tự:
         ├── run_lightgcn.sh    (sim_type=none)
         ├── run_img_only.sh    (sim_type=img_only)
         ├── run_multimodal.sh  (sim_type=multimodal)
         └── run_tfidf.sh       (sim_type=tfidf)
```
