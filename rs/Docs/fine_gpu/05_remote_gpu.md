# 05 — Remote GPU Workflow (RunPod / Vast.ai)

Hướng dẫn thuê GPU cloud và chạy toàn bộ pipeline trên đó qua SSH.

---

## 1. Chọn template

Khi tạo pod trên RunPod / Vast.ai, chọn:

```
pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
```

> Image này đã có PyTorch 2.9.1 + CUDA 12.8. **Không cần cài lại torch.**

---

## 2. Kết nối SSH

```bash
ssh root@<pod-ip> -p <port>
```

Kiểm tra nhanh:

```bash
nvidia-smi          # xem GPU, VRAM
python3 --version   # kiểm tra python (nên là 3.10.x)
```

---

## 3. Cài công cụ cần thiết

```bash
apt update && apt install -y git tmux curl gcc python3-dev
```

> `gcc` và `python3-dev` cần thiết cho Triton (unsloth kernel) — thiếu sẽ lỗi khi training.

Cài `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

---

## 4. Clone repo

Với repo **private** — dùng Personal Access Token (PAT):

```bash
git clone https://<username>:<YOUR_PAT>@github.com/<org>/<repo>.git
cd <repo>
```

> **Bảo mật:** Sau khi clone xong, xóa token khỏi remote URL:
> ```bash
> git remote set-url origin https://github.com/<org>/<repo>.git
> ```

Tạo `.env` từ template:

```bash
cp .env.example .env
nano .env
```

Các biến cần điền:

```
PROJECT_NAME=legal-slm-sft-v1
HF_TOKEN=hf_...              # huggingface.co/settings/tokens (Write)
HF_REPO_ID=HoangVuSnape/qwen3-1.7b-legal-sft-v1
WANDB_API_KEY=...            # wandb.ai/authorize
WANDB_ENTITY=<team-entity>
WANDB_PROJECT=legal-slm-finetune
DATA_DIR=./data
OUTPUT_DIR=./outputs
CACHE_DIR=./data/cache
```

---

## 5. Cài dependencies

```bash
# Cài vào system Python — torch đã có sẵn trong image, không cài lại
uv pip install --system -e ".[docker]"
uv pip install --system unsloth
```

> **Không dùng `uv sync`** — nó tạo `.venv` riêng với torch CUDA 12.4 thay vì dùng torch 2.9.1 có sẵn.

---

## 6. Đăng nhập HuggingFace & wandb

```bash
# HuggingFace — dùng token Write từ huggingface.co/settings/tokens
huggingface-cli login

# wandb — dùng API key từ wandb.ai/authorize
wandb login
```

Kiểm tra token HF hợp lệ:

```bash
python3 -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

---

## 7. Chạy training

```bash
chmod +x run_all/*.sh test/*.sh

# Smoke test trước (3 steps, ~1 phút)
./test/test_sft.sh

# Train thật
source .env
./run_all/sft_docker.sh
```

Hoặc chạy thẳng:

```bash
PYTHONPATH=. python3 src/app/train_sft.py --config configs/qwen3_1b7_legal.yaml
```

---

## 8. Chạy trong tmux (bắt buộc khi train lâu)

```bash
tmux new -s train

source .env
./run_all/sft_docker.sh

# Detach: Ctrl+B, rồi D
# Attach lại sau khi reconnect SSH:
tmux attach -t train
```

---

## 9. Theo dõi GPU

```bash
watch -n 2 nvidia-smi
```

| Chỉ số | Mục tiêu |
|---|---|
| GPU-Util | >80% khi training |
| Memory-Usage | <22GB (để buffer) |
| Temp | <85°C |

---

## 10. Push model lên HuggingFace

Sau khi training xong, adapter tự push nếu `.env` có `HF_REPO_ID`. Hoặc push thủ công:

```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='<HF_TOKEN>')
api.create_repo(repo_id='HoangVuSnape/qwen3-1.7b-legal-sft-v1', repo_type='model', exist_ok=True, private=True)
api.upload_folder(
    folder_path='outputs/legal-slm-sft-v1/adapter_final',
    repo_id='HoangVuSnape/qwen3-1.7b-legal-sft-v1',
    repo_type='model',
    commit_message='Upload adapter: legal-slm-sft-v1',
)
print('Done!')
"
```

---

## 11. Chạy evaluation

```bash
PYTHONPATH=. python3 scripts/eval.py \
    --config configs/qwen3_1b7_legal.yaml \
    --adapter outputs/legal-slm-sft-v1/adapter_final \
    --test_dir data/processed/sft/test \
    --output_dir outputs/evals/legal-slm-sft-v1 \
    --batch_size 32
```

Kết quả lưu tại `outputs/evals/legal-slm-sft-v1/summary.json`.

---

## Tóm tắt lệnh theo thứ tự

```bash
# 1. SSH
ssh root@<pod-ip> -p <port>

# 2. Cài tools
apt update && apt install -y git tmux curl gcc python3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

# 3. Clone & setup
git clone https://<user>:<PAT>@github.com/<org>/<repo>.git && cd <repo>
git remote set-url origin https://github.com/<org>/<repo>.git
cp .env.example .env && nano .env

# 4. Install
uv pip install --system -e ".[docker]"
uv pip install --system unsloth

# 5. Login
huggingface-cli login
wandb login

# 6. Test + Train trong tmux
tmux new -s train
chmod +x run_all/*.sh test/*.sh
./test/test_sft.sh && source .env && ./run_all/sft_docker.sh
# Ctrl+B, D để detach
```
