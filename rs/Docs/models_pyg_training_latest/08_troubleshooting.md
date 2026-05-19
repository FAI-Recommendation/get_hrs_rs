# 08 — Troubleshooting: Lỗi thực tế và cách fix

---

## Checklist nhanh

| Triệu chứng | Kiểm tra |
|---|---|
| `No module named 'utility'` | `cd rs/lightgcn_pyg` trước khi chạy |
| `No module named 'torch_sparse'` | Cài lại với wheel đúng version: `-f https://data.pyg.org/whl/torch-X.X.X+cuXXX.html` |
| `Permission denied` .sh | `chmod +x scripts/test/*.sh scripts/run_all/*.sh` |
| `python: not found` | Dùng `python3` hoặc tạo symlink |
| `triton C compiler` | `apt install -y gcc python3-dev` |
| CUDA OOM | Giảm `--batch_size` xuống 512 hoặc 256 |
| Adjacency build chậm | Bình thường lần đầu, sẽ cache sau. Đợi 5-15 phút. |
| Recall không tăng | Kiểm tra data density, hạ cosine threshold xuống 0.3 |
| `uv sync` sai torch | `rm -rf .venv`, dùng `uv pip install --system` |
| `requires-python` lỗi | Set `requires-python = ">=3.10"` trong pyproject.toml |

---

## 1. Python version không tương thích

**Lỗi:**
```
requires-python >=3.11, but pod có Python 3.10.12
```

**Fix:**
```toml
# pyproject.toml
requires-python = ">=3.10"
```

---

## 2. `uv sync` tạo `.venv` sai torch version

**Lỗi:**
```
torch 2.6.0+cu124 được cài vào .venv thay vì torch 2.1.2+cu121 có sẵn
ModuleNotFoundError hoặc AttributeError khi import torch_sparse
```

**Fix:**
```bash
# ĐÚNG — cài thẳng vào system Python của image
uv pip install --system -e ".[docker]"
uv pip install --system torch-geometric
uv pip install --system torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# SAI — KHÔNG dùng cái này trên Docker pod
# uv sync
```

Nếu đã lỡ chạy `uv sync`:
```bash
rm -rf .venv
```

---

## 3. `ModuleNotFoundError: No module named 'utility'`

**Nguyên nhân:** Chạy `python3 train.py` từ thư mục sai.

**Fix:**
```bash
# Phải cd vào lightgcn_pyg trước
cd rs/lightgcn_pyg
python3 train.py ...

# Hoặc dùng PYTHONPATH
PYTHONPATH=rs/lightgcn_pyg python3 rs/lightgcn_pyg/train.py ...
```

---

## 4. `ModuleNotFoundError: No module named 'torch_sparse'`

**Fix:**
```bash
# Kiểm tra torch version
python3 -c "import torch; print(torch.__version__)"
# Ví dụ: 2.1.2+cu121

# Cài torch-sparse đúng wheel
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

Các URL wheel phổ biến:

| torch + CUDA | URL |
|---|---|
| 2.1.2 + cu121 | `https://data.pyg.org/whl/torch-2.1.2+cu121.html` |
| 2.9.0 + cu128 | `https://data.pyg.org/whl/torch-2.9.0+cu128.html` |

---

## 5. Permission denied khi chạy shell script

```bash
chmod +x scripts/test/*.sh scripts/run_all/*.sh
```

---

## 6. `python` command not found

```bash
# Tạo symlink
ln -s /usr/bin/python3 /usr/local/bin/python
```

---

## 7. Triton lỗi không tìm thấy C compiler

```bash
apt update && apt install -y gcc python3-dev
```

---

## 8. CUDA out of memory

**Fix 1 — Giảm batch_size:**
```bash
python3 train.py ... --batch_size 512
```

**Fix 2 — Giảm embed_size:**
```bash
python3 train.py ... --embed_size 32 --layer_size "[32,32,32]"
```

**Fix 3 — Kiểm tra VRAM:**
```bash
nvidia-smi
# VRAM    Khuyến nghị batch_size
# 6 GB  → 256-512
# 12 GB → 512-1024
# 24 GB → 1024-4096
```

---

## 9. Adjacency matrix build rất chậm lần đầu

**Đây là bình thường.** Chỉ build 1 lần, sau đó cache thành file `.npz`:

```
get10k_data/clip_10k_sample/
├── s_interaction_adj_mat.npz         # nhanh (~1s)
├── s_tfidf_item_similarity_adj_mat.npz   # nhanh (~5s)
├── s_bert_item_similarity_adj_mat.npz    # CHẬM — BERT inference (~5-15 phút)
├── s_multimodal_similarity_adj_mat.npz   # nhanh sau khi có BERT cache
└── s_img_similarity_adj_mat.npz      # nhanh (~10s)
```

Nếu cần rebuild sau khi đổi data:
```bash
rm get10k_data/clip_10k_sample/s_*.npz
```

---

## 10. `ValueError: feature3 column not found` hoặc image embedding lỗi

**Kiểm tra:**
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('get10k_data/clip_10k_sample/items_features.csv')
print(df.columns.tolist())
print(df['feature3'].iloc[0][:100])
print(df['feature3'].isna().sum(), 'rows NaN')
"
```

`parse_vector_string()` hỗ trợ các format:
```
"0.123 -0.456 0.789"      # space-separated → OK
"0.123,-0.456,0.789"      # comma-separated → OK
"[0.123, -0.456, 0.789]"  # bracket + comma → OK
```

---

## 11. Recall không tăng sau nhiều epochs

**Kiểm tra 1 — Dữ liệu:**
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

Mỗi user nên có ít nhất **3-5 interactions** trong train set.

**Kiểm tra 2 — Adjacency matrix có build đúng không:**
```bash
python3 -c "
import scipy.sparse as sp
mat = sp.load_npz('get10k_data/clip_10k_sample/s_img_similarity_adj_mat.npz')
print('Shape:', mat.shape)
print('NNZ:', mat.nnz)
print('Density:', mat.nnz / (mat.shape[0] * mat.shape[1]))
"
```

Nếu `NNZ = 0` → similarity matrix rỗng → hạ threshold:

```python
# Trong utility/load_data.py
# Đổi:  sim_matrix[sim_matrix < 0.5] = 0
# Thành:
sim_matrix[sim_matrix < 0.3] = 0  # hạ ngưỡng để có nhiều edges hơn
```

**Fix khác:**
```bash
--epoch 2000              # tăng epoch
--lr 0.0005              # giảm learning rate
--early_stop_steps 10    # tăng patience
```

---

## 12. Training mất quá lâu trên CPU

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Nếu `False`:
1. Torch được cài phiên bản CPU → re-install với CUDA
2. GPU driver chưa được load → kiểm tra `nvidia-smi`

Trên máy local không có GPU: dùng `--gpu_id -1` chỉ để test logic.
