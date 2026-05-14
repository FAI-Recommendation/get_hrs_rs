# Troubleshooting — Loi thuc te da gap

Tong hop cac loi phat sinh khi setup va chay CombiGCN pipeline tren **RunPod / Vast.ai**,
cung cach fix tuong ung.

---

## 1. Python version khong tuong thich

**Loi:**
```
requires-python >=3.11, but pod co Python 3.10.12
uv khong cai duoc package
```

**Nguyen nhan:** `pyproject.toml` khai bao Python 3.11+ trong khi Docker image `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` di kem Python 3.10.

**Fix:**
```toml
# pyproject.toml
requires-python = ">=3.10"
```

> Khong nang Python vi image khong co san 3.11+. Phai ha yeu cau xuong 3.10.

---

## 2. `uv sync` tao `.venv` sai torch version

**Loi:**
```
torch 2.6.0+cu124 duoc cai vao .venv thay vi dung torch 2.1.2+cu121 co san
ModuleNotFoundError hoac AttributeError khi import torch_sparse
```

**Nguyen nhan:** `uv sync` luon tao `.venv` moi va resolve torch tu PyPI — tai ve CUDA 12.4 thay vi dung torch co san trong image.

**Fix:**
```bash
# DUNG — cai thang vao system Python cua image
uv pip install --system -e ".[docker]"
uv pip install --system torch-geometric
uv pip install --system torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# SAI — KHONG dung cai nay tren Docker pod
# uv sync
```

Neu da lo chay `uv sync`, xoa di:
```bash
rm -rf .venv
```

---

## 3. `ModuleNotFoundError: No module named 'utility'`

**Loi:**
```
ModuleNotFoundError: No module named 'utility'
```

**Nguyen nhan:** Chay `python3 train.py` tu thu muc sai, hoac khong set `PYTHONPATH`.

**Fix — Chay tu dung thu muc:**
```bash
# Phai cd vao lightgcn_pyg truoc
cd rs/lightgcn_pyg
python3 train.py ...

# Hoac dung PYTHONPATH
PYTHONPATH=rs/lightgcn_pyg python3 rs/lightgcn_pyg/train.py ...
```

> Tat ca cac shell scripts da set `cd "$(dirname "$0")/../.."` nen chay scripts thi khong bi loi nay.

---

## 4. `ModuleNotFoundError: No module named 'torch_sparse'`

**Loi:**
```
ModuleNotFoundError: No module named 'torch_sparse'
```

**Nguyen nhan:** `torch-sparse` can duoc cai voi wheel match chinh xac voi version torch + CUDA.

**Fix:**
```bash
# Kiem tra torch version truoc
python3 -c "import torch; print(torch.__version__)"
# Vi du: 2.1.2+cu121

# Cai torch-sparse dung wheel
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# Hoac dung uv
uv pip install --system torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

Cac URL wheel pho bien:

| torch + CUDA | URL |
|---|---|
| 2.1.2 + cu121 | `https://data.pyg.org/whl/torch-2.1.2+cu121.html` |
| 2.2.0 + cu121 | `https://data.pyg.org/whl/torch-2.2.0+cu121.html` |
| 2.3.0 + cu121 | `https://data.pyg.org/whl/torch-2.3.0+cu121.html` |

---

## 5. Permission denied khi chay shell script

**Loi:**
```
bash: ./scripts/run_all/run_all.sh: Permission denied
```

**Nguyen nhan:** File `.sh` sau khi clone tu git tren Linux khong co execute permission.

**Fix:**
```bash
chmod +x scripts/test/*.sh scripts/run_all/*.sh
```

---

## 6. `python` command not found

**Loi:**
```
bash: python: command not found
```

**Nguyen nhan:** Docker image chi co `python3`, khong co symlink `python`.

**Fix:** Doi `python` thanh `python3` trong moi lenh, hoac tao symlink:
```bash
ln -s /usr/bin/python3 /usr/local/bin/python
```

---

## 7. Triton loi khong tim thay C compiler

**Loi:**
```
triton RuntimeError: Failed to find C compiler. Please specify via CC environment variable
```

**Nguyen nhan:** Docker image khong co `gcc` mac dinh.

**Fix:**
```bash
apt update && apt install -y gcc python3-dev
```

---

## 8. CUDA out of memory

**Loi:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate ... (GPU 0; ... total capacity; ... already allocated)
```

**Nguyen nhan:** `batch_size` qua lon so voi VRAM, hoac adjacency matrix qua lon.

**Fix 1 — Giam batch_size:**
```bash
python3 train.py ... --batch_size 512
# Hoac
python3 train.py ... --batch_size 256
```

**Fix 2 — Giam embed_size:**
```bash
python3 train.py ... --embed_size 32 --layer_size "[32,32,32]"
```

**Fix 3 — Kiem tra VRAM truoc:**
```bash
nvidia-smi
# VRAM    Khuyen nghi batch_size
# 6 GB  → 256-512
# 12 GB → 512-1024
# 24 GB → 1024-4096
```

---

## 9. Adjacency matrix build rat cham lan dau

**Van de:** Lan dau chay mat 5-15 phut de build 8 adjacency matrices, khong thay progress.

**Nguyen nhan:** Cac matrices nhu `bert_item_similarity` can chay BERT inference tren toan bo item catalog.

**Day la binh thuong.** Chi build 1 lan, sau do cache thanh file `.npz`:

```
get10k_data/clip_10k_sample/
├── s_interaction_adj_mat.npz         # nhanh (~1s)
├── s_tfidf_item_similarity_adj_mat.npz   # nhanh (~5s)
├── s_bert_item_similarity_adj_mat.npz    # CHAM - BERT inference (~5-15 phut)
├── s_multimodal_similarity_adj_mat.npz   # nhanh sau khi co BERT cache
└── s_img_similarity_adj_mat.npz      # nhanh (~10s)
```

Lan chay tiep theo se load tu cache trong vai giay.

**Neu can rebuild** (sau khi doi data):
```bash
rm get10k_data/clip_10k_sample/s_*.npz
```

---

## 10. `ValueError: feature3 column not found` hoac image embedding loi

**Loi:**
```
KeyError: 'feature3'
# hoac
ValueError: could not convert string to float
```

**Nguyen nhan 1:** File `items_features.csv` khong co cot `feature3`.

**Nguyen nhan 2:** Gia tri `feature3` la chuoi rong hoac format sai.

**Kiem tra:**
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('get10k_data/clip_10k_sample/items_features.csv')
print(df.columns.tolist())
print(df['feature3'].iloc[0][:100])  # xem 100 ky tu dau
print(df['feature3'].isna().sum(), 'rows NaN')
"
```

**Fix — parse_vector_string() ho tro ca space va comma delimiter**, nen format nao cung duoc:
```
"0.123 -0.456 0.789"      # space-separated → OK
"0.123,-0.456,0.789"      # comma-separated → OK
"[0.123, -0.456, 0.789]"  # bracket + comma → OK
```

---

## 11. Recall khong tang sau nhieu epochs

**Van de:** Model train duoc nhung recall@K gam nhu khong cai thien (o muc rat thap).

**Kiem tra 1 — Du lieu:**
```bash
python3 -c "
from utility.load_data import Data
data = Data('get10k_data/clip_10k_sample/', batch_size=1024)
print('n_users:', data.n_users)
print('n_items:', data.n_items)
print('n_train:', data.n_train)
print('avg interactions per user:', data.n_train / data.n_users)
"
```

Moi user nen co it nhat **3-5 interactions** trong train set. Neu trung binh < 2, du lieu qua tha.

**Kiem tra 2 — Adjacency matrix co build dung khong:**
```bash
python3 -c "
import scipy.sparse as sp
import numpy as np
mat = sp.load_npz('get10k_data/clip_10k_sample/s_img_similarity_adj_mat.npz')
print('Shape:', mat.shape)
print('NNZ:', mat.nnz)
print('Density:', mat.nnz / (mat.shape[0] * mat.shape[1]))
"
```

Neu `NNZ = 0` → similarity matrix rong → tat ca items co cosine similarity < 0.5 → thu ha nguong trong `load_data.py`.

**Fix — Ha threshold cosine similarity:**
```python
# Trong utility/load_data.py
# Tim dong: sim_matrix[sim_matrix < 0.5] = 0
# Doi thanh:
sim_matrix[sim_matrix < 0.3] = 0  # ha nguong de co nhieu edges hon
```

---

## 12. Training mat qua lau tren CPU (khong co GPU)

**Van de:** Training cham gam nhu dung.

**Kiem tra:**
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Neu `False`, co the:
1. Torch duoc cai phien ban CPU → re-install voi CUDA
2. GPU driver chua duoc load → kiem tra `nvidia-smi`

**Tren may local khong co GPU:** Dung `--gpu_id -1` de chay tren CPU (chi de test logic, khong train that).

---

## Tom tat nhanh — Checklist khi gap loi

| Trieu chung | Kiem tra |
|---|---|
| `No module named 'utility'` | `cd rs/lightgcn_pyg` truoc khi chay |
| `No module named 'torch_sparse'` | Cai lai voi wheel dung version: `-f https://data.pyg.org/whl/torch-X.X.X+cuXXX.html` |
| `Permission denied` .sh | `chmod +x scripts/test/*.sh scripts/run_all/*.sh` |
| `python: not found` | Dung `python3` hoac tao symlink |
| `triton C compiler` | `apt install -y gcc python3-dev` |
| CUDA OOM | Giam `--batch_size` xuong 512 hoac 256 |
| Adjacency build cham | Binh thuong lan dau, se cache sau. Doi 5-15 phut. |
| Recall khong tang | Kiem tra data density, ha cosine threshold xuong 0.3 |
| `uv sync` sai torch | `rm -rf .venv`, dung `uv pip install --system` |
| `requires-python` loi | Set `requires-python = ">=3.10"` trong pyproject.toml |
