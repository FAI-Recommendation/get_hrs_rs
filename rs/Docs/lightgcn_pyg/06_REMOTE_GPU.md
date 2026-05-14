# Remote GPU Workflow (RunPod / Vast.ai)

Huong dan thue GPU cloud va chay toan bo CombiGCN pipeline qua SSH.

---

## 1. Chon template

Khi tao pod tren RunPod / Vast.ai, chon:

```
pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
```

> Image nay da co PyTorch 2.1.2 + CUDA 12.1. **Khong can cai lai torch.**

GPU khuyen nghi:

| GPU | VRAM | Thoi gian train (img_only) | Chi phi uoc tinh |
|---|---|---|---|
| RTX 3090 | 24 GB | ~30-45 phut | ~$0.3/h |
| RTX 4090 | 24 GB | ~20-30 phut | ~$0.5/h |
| A5000 | 24 GB | ~35-50 phut | ~$0.4/h |
| A100 | 40/80 GB | ~15-25 phut | ~$1.5/h |

---

## 2. Ket noi SSH

```bash
ssh root@<pod-ip> -p <port>
```

Kiem tra nhanh:

```bash
nvidia-smi          # xem GPU, VRAM
python3 --version   # nen la 3.10.x
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. Cai cong cu can thiet

```bash
apt update && apt install -y git tmux curl gcc python3-dev
```

> `gcc` va `python3-dev` can thiet cho cac extension compile (torch-sparse).

Cai `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

---

## 4. Clone repo

Voi repo **private** — dung Personal Access Token (PAT):

```bash
git clone https://<username>:<YOUR_PAT>@github.com/<org>/<repo>.git
cd <repo>
```

> **Bao mat:** Sau khi clone xong, xoa token khoi remote URL:
> ```bash
> git remote set-url origin https://github.com/<org>/<repo>.git
> ```

Tao `.env` tu template:

```bash
cp .env.example .env
nano .env
```

Cap nhat bien trong `.env`:

```
DATA_DIR=./get10k_data
OUTPUT_DIR=./rs/lightgcn_pyg/output
CUDA_VISIBLE_DEVICES=0
GPU_ID=0
```

---

## 5. Cai dependencies

```bash
# Cai vao system Python — torch da co san trong image, khong cai lai
uv pip install --system -e ".[docker]"

# Cai torch-geometric va torch-sparse (can match voi torch version trong image)
# Cho torch 2.1.2 + CUDA 12.1:
uv pip install --system torch-geometric
uv pip install --system torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

> **Khong dung `uv sync`** — no tao `.venv` moi voi torch CUDA 12.4 thay vi dung torch 2.1.2 co san.

Kiem tra sau khi cai:

```bash
python3 -c "
import torch
import torch_geometric
import torch_sparse
print('torch:', torch.__version__)
print('torch_geometric:', torch_geometric.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
"
```

---

## 6. Dang nhap wandb + HuggingFace (optional)

### Wandb (xem eval metrics online)

```bash
wandb login
# Nhap API key tu: wandb.ai/authorize
```

Kiem tra:
```bash
python3 -c "import wandb; wandb.login(); print('wandb OK')"
```

### HuggingFace (save best_model.pt len Hub)

```bash
# Dung token Write tu: huggingface.co/settings/tokens
huggingface-cli login
```

Kiem tra token hop le:
```bash
python3 -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

---

## 8. Chay training

```bash
cd rs/lightgcn_pyg
chmod +x scripts/test/*.sh scripts/run_all/*.sh

# Smoke test truoc (~2 phut, 20 epochs)
bash scripts/test/test_lightgcn.sh

# Train that — 1 variant
bash scripts/run_all/run_img_only.sh

# Hoac train ca 4 variants lien tiep
bash scripts/run_all/run_all.sh
```

Chay voi **wandb + HF** (full options):

```bash
set -a; source .env; set +a   # export tat ca bien trong .env

python3 train.py \
    --data_path ../../get10k_data/clip_10k_sample \
    --sim_type img_only \
    --epoch 1000 \
    --eval_interval 10 \
    --early_stop_steps 5 \
    --save_flag 1 \
    --gpu_id 0 \
    --use_wandb 1 \
    --wandb_project combigcn-rs \
    --wandb_entity $WANDB_ENTITY \
    --use_hf 1 \
    --hf_repo_id $HF_REPO_ID
```

> **Luu y**: `set -a; source .env; set +a` export tat ca bien ra environment (khac `source .env` thong thuong chi load vao shell).

---

## 9. Chay trong tmux (bat buoc khi train lau)

```bash
tmux new -s train

cd rs/lightgcn_pyg
bash scripts/run_all/run_all.sh

# Detach: Ctrl+B, roi D
# Attach lai sau khi reconnect SSH:
tmux attach -t train
```

---

## 10. Theo doi GPU

```bash
watch -n 2 nvidia-smi
```

| Chi so | Muc tieu |
|---|---|
| GPU-Util | >60% khi training (GCN mat mul) |
| Memory-Usage | <16 GB voi batch_size=1024 |
| Temp | <85°C |

Neu GPU-Util thap (<30%), thu tang `--batch_size`:

```bash
--batch_size 2048    # ~4 GB VRAM
--batch_size 4096    # ~8 GB VRAM
```

---

## 11. Theo doi training qua TensorBoard + wandb

Tren may local, tao SSH tunnel de xem TensorBoard:

```bash
# Tren may local:
ssh -L 6006:localhost:6006 root@<pod-ip> -p <port>

# Tren pod (trong tmux session khac):
tensorboard --logdir rs/lightgcn_pyg/tensorboard/ --host 0.0.0.0
```

Mo browser tai `http://localhost:6006`.

Wandb tu dong log len `wandb.ai` — khong can SSH tunnel, xem tu bat ky dau.

---

## 10. Lay ket qua ve may local

Sau khi training xong:

```bash
# Tren may local — copy weights va results
scp -P <port> -r root@<pod-ip>:<repo>/rs/lightgcn_pyg/weights ./
scp -P <port> -r root@<pod-ip>:<repo>/rs/lightgcn_pyg/output ./
```

Hoac dung `rsync` de dong bo nhanh hon:

```bash
rsync -avz -e "ssh -p <port>" \
    root@<pod-ip>:<repo>/rs/lightgcn_pyg/weights \
    root@<pod-ip>:<repo>/rs/lightgcn_pyg/output \
    ./rs/lightgcn_pyg/
```

---

## 11. Load model da train

```python
import torch
from model import CombiGCN

checkpoint = torch.load('weights/img_only/layers_3_dim_64/lr_0.001_reg_1e-05/best_model.pt')
model = CombiGCN(
    n_users=checkpoint['args']['n_users'],
    n_items=checkpoint['args']['n_items'],
    embed_dim=checkpoint['args']['embed_size'],
    n_layers=len(checkpoint['args']['layer_size']),
    decay=checkpoint['args']['regs'][0],
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Tom tat lenh theo thu tu

```bash
# 1. SSH
ssh root@<pod-ip> -p <port>

# 2. Cai tools
apt update && apt install -y git tmux curl gcc python3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

# 3. Clone & setup
git clone https://<user>:<PAT>@github.com/<org>/<repo>.git && cd <repo>
git remote set-url origin https://github.com/<org>/<repo>.git
cp .env.example .env && nano .env

# 4. Install
uv pip install --system -e ".[docker]"
uv pip install --system torch-geometric
uv pip install --system torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# 5. Verify
python3 -c "import torch; import torch_sparse; print(torch.cuda.is_available())"

# 6. Smoke test + train trong tmux
tmux new -s train
cd rs/lightgcn_pyg
chmod +x scripts/test/*.sh scripts/run_all/*.sh
bash scripts/test/test_lightgcn.sh && bash scripts/run_all/run_all.sh
# Ctrl+B, D de detach
```
