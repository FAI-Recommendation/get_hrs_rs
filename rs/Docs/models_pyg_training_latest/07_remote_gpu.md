# 07 — Remote GPU Workflow (RunPod / Vast.ai)

---

## Tóm tắt các bước

Clone repo → chỉnh `.env` → chạy `setup_env.sh` → smoke test → train trong tmux.

---

## 1. Chọn template

```
pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
```

Image này đã có PyTorch 2.9.1 + CUDA 12.8. **Không cần cài lại torch.**

GPU khuyến nghị:

| GPU | VRAM | Thời gian train (img_only) | Chi phí |
|---|---|---|---|
| RTX 3090 | 24 GB | ~30-45 phút | ~$0.3/h |
| RTX 4090 | 24 GB | ~20-30 phút | ~$0.5/h |
| A5000 | 24 GB | ~35-50 phút | ~$0.4/h |
| A100 | 40/80 GB | ~15-25 phút | ~$1.5/h |

---

## 2. Kết nối SSH

```bash
ssh root@<pod-ip> -p <port>

# Kiểm tra nhanh
nvidia-smi
python3 --version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. Cài công cụ cần thiết

```bash
sudo apt update && sudo apt install -y git tmux curl build-essential python3-dev cmake libomp-dev

# Cài uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

---

## 4. Clone repo

```bash
# Với repo private — dùng PAT
git clone https://<username>:<YOUR_PAT>@github.com/<org>/<repo>.git
cd <repo>

# Xóa token khỏi remote URL sau khi clone
git remote set-url origin https://github.com/<org>/<repo>.git

# Tạo .env
cp .env.example .env
nano .env
```

Config trong `.env`:

```
DATA_DIR=./get10k_data
OUTPUT_DIR=./rs/lightgcn_pyg/output
CUDA_VISIBLE_DEVICES=0
GPU_ID=0
MULTIMODAL_METHOD=late_fusion
MULTIMODAL_PCA_COMPONENTS=256
```

---

## 5. Cài dependencies (1 lệnh)

```bash
chmod +x rs/lightgcn_pyg/scripts/setup_env.sh
bash rs/lightgcn_pyg/scripts/setup_env.sh
```

Script tự động:
- Detect CUDA version từ `nvidia-smi`
- Cài torch nếu chưa có (chọn đúng version cho CUDA đó)
- Cài torch-geometric + torch-sparse + pyg_lib
- Cài tất cả project dependencies
- Verify imports cuối cùng

Nếu muốn pin version cụ thể:
```bash
TORCH_VERSION=2.9.0 CUDA_TAG=cu128 bash rs/lightgcn_pyg/scripts/setup_env.sh

# Hoặc override wheel index
TORCH_WHL_URL=https://data.pyg.org/whl/torch-2.1.2+cu121.html bash setup_env.sh
```

Cài thủ công (nếu cần):
```bash
uv pip install --system -e ".[docker]"
uv pip install --system torch-geometric
uv pip install --system pyg_lib torch_scatter torch_sparse torch \
    -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

**Không dùng `uv sync`** — nó tạo `.venv` mới tách biệt, không dùng Python của image.

Kiểm tra:
```bash
python3 -c "import torch, torch_geometric, torch_sparse; print(torch.__version__, torch.cuda.is_available())"
```

---

## 6. Đăng nhập wandb + HuggingFace (optional)

```bash
# wandb
wandb login  # nhập API key từ: wandb.ai/authorize

# HuggingFace
huggingface-cli login  # dùng token Write từ: huggingface.co/settings/tokens
```

---

## 7. Smoke test + Training trong tmux

```bash
tmux new -s train

cd rs/lightgcn_pyg
chmod +x scripts/test/*.sh scripts/run_all/*.sh

# Smoke test (~2 phút, 20 epochs)
bash scripts/test/test_lightgcn.sh

# Train thật — 1 variant
bash scripts/run_all/run_img_only.sh

# Hoặc train cả 4 variants liên tiếp
bash scripts/run_all/run_all.sh

# Detach: Ctrl+B rồi D
# Attach lại sau khi reconnect:
tmux attach -t train
```

Chạy với wandb + HF (full options):

```bash
set -a; source .env; set +a   # export tất cả biến trong .env

python3 train.py \
    --data_path ../../get10k_data/clip_10k_sample \
    --sim_type img_only \
    --epoch 1000 \
    --eval_interval 40 \
    --early_stop_steps 0 \
    --save_flag 1 \
    --gpu_id 0 \
    --use_wandb 1 \
    --wandb_project combigcn-rs \
    --wandb_entity $WANDB_ENTITY \
    --use_hf 1 \
    --hf_repo_id $HF_REPO_ID
```

---

## 8. Theo dõi GPU

```bash
watch -n 2 nvidia-smi
```

| Chỉ số | Mục tiêu |
|---|---|
| GPU-Util | >60% khi training |
| Memory-Usage | <16 GB với batch_size=1024 |
| Temp | <85°C |

Nếu GPU-Util thấp (<30%), tăng `--batch_size 2048` hoặc `4096`.

---

## 9. Theo dõi qua TensorBoard (SSH tunnel)

```bash
# Trên máy local:
ssh -L 6006:localhost:6006 root@<pod-ip> -p <port>

# Trên pod (trong tmux session khác):
tensorboard --logdir rs/lightgcn_pyg/tensorboard/ --host 0.0.0.0
```

Mở `http://localhost:6006`. Wandb tự động log lên `wandb.ai` — không cần SSH tunnel.

---

## 10. Lấy kết quả về máy local

```bash
# scp
scp -P <port> -r root@<pod-ip>:<repo>/rs/lightgcn_pyg/weights ./
scp -P <port> -r root@<pod-ip>:<repo>/rs/lightgcn_pyg/output ./

# rsync (nhanh hơn)
rsync -avz -e "ssh -p <port>" \
    root@<pod-ip>:<repo>/rs/lightgcn_pyg/weights \
    root@<pod-ip>:<repo>/rs/lightgcn_pyg/output \
    ./rs/lightgcn_pyg/
```

---

## 11. Load model đã train

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

## Tóm tắt lệnh theo thứ tự

```bash
# 1. SSH
ssh root@<pod-ip> -p <port>

# 2. Clone & setup
git clone https://<user>:<PAT>@github.com/<org>/<repo>.git && cd <repo>
git remote set-url origin https://github.com/<org>/<repo>.git
cp .env.example .env && nano .env

# 3. Cài dependencies
bash rs/lightgcn_pyg/scripts/setup_env.sh

# 4. (Optional) Đăng nhập wandb + HF
wandb login
huggingface-cli login

# 5. Smoke test + train trong tmux
tmux new -s train
cd rs/lightgcn_pyg
chmod +x scripts/test/*.sh scripts/run_all/*.sh
bash scripts/test/test_lightgcn.sh && bash scripts/run_all/run_all.sh
# Ctrl+B, D để detach
```
