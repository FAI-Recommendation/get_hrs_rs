# Scripts — Huong dan chay

## 1. Cau truc

```
scripts/
├── test/                    # Test nhanh (20 epochs)
│   ├── test_all.sh          # Chay ca 4 variants lien tiep
│   ├── test_lightgcn.sh     # LightGCN thuan (--sim_type none)
│   ├── test_img_only.sh     # CombiGCN + Image Only
│   ├── test_multimodal.sh   # CombiGCN + Multimodal
│   └── test_tfidf.sh        # CombiGCN + TF-IDF
│
└── run_all/                 # Full training (1000 epochs)
    ├── run_all.sh           # Chay ca 4 variants lien tiep
    ├── run_lightgcn.sh      # LightGCN thuan
    ├── run_img_only.sh      # CombiGCN + Image Only
    ├── run_multimodal.sh    # CombiGCN + Multimodal
    └── run_tfidf.sh         # CombiGCN + TF-IDF
```

## 2. Test Scripts (20 epochs)

### Muc dich

Verify toan bo pipeline hoat dong truoc khi chay full training:
- Khong luu model (`--save_flag 0`)
- Chi 20 epochs, eval moi 10 epoch → 2 lan eval
- Ks nho (`[1,5,10,20]`) de chay nhanh
- Early stop sau 3 intervals

### Chay

```bash
cd rs/lightgcn_pyg

# Chay tat ca 4 variants
bash scripts/test/test_all.sh

# Hoac chay tung variant
bash scripts/test/test_lightgcn.sh      # ~1-2 phut
bash scripts/test/test_img_only.sh       # ~2-3 phut
bash scripts/test/test_multimodal.sh     # ~2-3 phut
bash scripts/test/test_tfidf.sh          # ~2-3 phut
```

### Tham so test

| Tham so | Gia tri | Ly do |
|---|---|---|
| `--epoch` | 20 | Chi can verify pipeline, khong can hoi tu |
| `--eval_interval` | 10 | Eval 2 lan: epoch 10 va 20 |
| `--early_stop_steps` | 3 | Dung som neu bi loi |
| `--Ks` | [1,5,10,20] | Bo K=50 de giam thoi gian eval |
| `--save_flag` | 0 | Khong luu weights |
| `--batch_size` | 1024 | Giong full training |

### Ket qua mong doi

Neu test pass, ban se thay output tuong tu:

```
Epoch 10/20 — eval on test set:
  precision=[0.xxx, 0.xxx, 0.xxx, 0.xxx]
  recall=[0.xxx, 0.xxx, 0.xxx, 0.xxx]
  ndcg=[0.xxx, 0.xxx, 0.xxx, 0.xxx]
  ...

Epoch 20/20 — eval on test set:
  precision=[0.xxx, 0.xxx, 0.xxx, 0.xxx]
  ...
```

**Luu y**: So lieu se rat thap vi chi train 20 epochs. Muc dich la verify khong bi crash, khong phai do hieu qua.

## 3. Full Training Scripts (1000 epochs)

### Muc dich

Train day du voi cac thiet lap tot nhat:
- Luu best model (`--save_flag 1`)
- 1000 epochs toi da, early stopping se dung khi hoi tu
- Ks day du (`[1,5,10,20,50]`)
- Ket qua luu ra file

### Chay

```bash
cd rs/lightgcn_pyg

# Chay tat ca 4 variants (tuan tu)
bash scripts/run_all/run_all.sh

# Hoac chay tung variant
bash scripts/run_all/run_lightgcn.sh     # ~30-60 phut
bash scripts/run_all/run_img_only.sh      # ~30-60 phut
bash scripts/run_all/run_multimodal.sh    # ~30-60 phut
bash scripts/run_all/run_tfidf.sh         # ~30-60 phut
```

### Tham so full training

| Tham so | Gia tri | Ly do |
|---|---|---|
| `--epoch` | 1000 | Du epochs de hoi tu |
| `--eval_interval` | 10 | Eval moi 10 epochs |
| `--early_stop_steps` | 5 | Dung sau 50 epochs khong cai thien (5 * 10) |
| `--Ks` | [1,5,10,20,50] | Day du K values de so sanh |
| `--save_flag` | 1 | Luu best model weights |
| `--weights_path` | weights/ | Thu muc luu weights |
| `--output_path` | output/ | Thu muc luu ket qua |
| `--batch_size` | 1024 | Can bang toc do va VRAM |

### Output

Sau khi chay xong, thu muc se co:

```
rs/lightgcn_pyg/
├── weights/
│   ├── none/layers_3_dim_64/lr_0.001_reg_1e-05/
│   │   └── best_model.pt
│   ├── img_only/layers_3_dim_64/lr_0.001_reg_1e-05/
│   │   └── best_model.pt
│   ├── multimodal/layers_3_dim_64/lr_0.001_reg_1e-05/
│   │   └── best_model.pt
│   └── tfidf/layers_3_dim_64/lr_0.001_reg_1e-05/
│       └── best_model.pt
│
├── output/
│   ├── none/combigcn.result
│   ├── img_only/combigcn.result
│   ├── multimodal/combigcn.result
│   └── tfidf/combigcn.result
│
└── tensorboard/
    ├── none_layers3_dim64_lr0.001_reg1e-05/
    ├── img_only_layers3_dim64_lr0.001_reg1e-05/
    ├── multimodal_layers3_dim64_lr0.001_reg1e-05/
    └── tfidf_layers3_dim64_lr0.001_reg1e-05/
```

## 4. So sanh 4 Variants

| Variant | `--sim_type` | File TF1 goc | Dac diem |
|---|---|---|---|
| LightGCN thuan | `none` | `LightGCN.py` | Chi bipartite graph, nhanh nhat |
| CombiGCN + Image | `img_only` | `LightGCN_only_img.py` | Them image similarity giua items |
| CombiGCN + Multimodal | `multimodal` | `LightGCN_bert_img.py` | alpha*text + (1-alpha)*image |
| CombiGCN + TF-IDF | `tfidf` | `LightGCN_tfidf_bert.py` | TF-IDF text similarity giua items |

### Thu tu nen chay

1. **`none`** (baseline) — de co so lieu co ban de so sanh
2. **`img_only`** — xem anh huong cua image features
3. **`multimodal`** — ket hop text + image
4. **`tfidf`** — chi text features

## 5. Tuy chinh tham so

### Thay doi data path

Sua `--data_path` trong tung file `.sh`:

```bash
--data_path ../../get10k_data/clip_10k_sample    # Mac dinh
--data_path ../../get10k_data/mobilenet_10k_sample  # Doi sang MobileNet data
--data_path /path/to/your/data                      # Duong dan tuy chinh
```

### Thay doi GPU

```bash
--gpu_id 0    # GPU dau tien (mac dinh)
--gpu_id 1    # GPU thu hai
--gpu_id -1   # CPU (cham, chi de test)
```

### Tang batch size (neu GPU du VRAM)

```bash
--batch_size 2048    # Nhanh hon, can ~4GB VRAM
--batch_size 4096    # Nhanh nhat, can ~8GB VRAM
```

### Thay doi so layers

```bash
--layer_size "[64,64]"          # 2 layers (it parameters hon)
--layer_size "[64,64,64]"       # 3 layers (mac dinh)
--layer_size "[64,64,64,64]"    # 4 layers (nhieu parameters hon)
```

## 6. TensorBoard

Theo doi training real-time:

```bash
tensorboard --logdir rs/lightgcn_pyg/tensorboard/
```

Mo browser tai `http://localhost:6006` de xem:
- **Scalars**: loss, recall, precision, ndcg, ... theo epoch
- **So sanh**: chon nhieu runs de so sanh cac variants

## 7. Troubleshooting

### Loi "ModuleNotFoundError: torch_geometric"

```bash
pip install torch-geometric torch-sparse
```

### Loi "CUDA out of memory"

Giam `--batch_size` xuong 512 hoac 256:

```bash
python train.py ... --batch_size 512
```

### Loi "FileNotFoundError: train.txt"

Kiem tra `--data_path` co dung thu muc chua `train.txt` va `test.txt` khong.

### Adjacency matrix build cham (lan dau)

Lan dau chay se mat vai phut de build 8 adjacency matrices. Sau do se cache thanh file `.npz` va load nhanh hon nhieu. Neu doi data, xoa cac file `s_*.npz` de rebuild:

```bash
rm clip_10k_sample/s_*.npz
```

### Test pass nhung full training khong cai thien

- Thu tang `--epoch` len 2000
- Thu giam `--lr` xuong 0.0005
- Thu tang `--early_stop_steps` len 10
- Kiem tra data co du tuong tac khong (moi user can it nhat 2-3 items)
