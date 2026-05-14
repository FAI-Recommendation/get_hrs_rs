# Troubleshooting — Lỗi thực tế đã gặp

Tổng hợp toàn bộ lỗi thực tế phát sinh khi setup và chạy pipeline trên **RunPod A5000 24GB**,
cùng cách fix tương ứng.

---

## 1. Python version không tương thích

**Lỗi:**
```
requires-python >=3.11, but pod có Python 3.10.12
uv không cài được package
```

**Nguyên nhân:** `pyproject.toml` và `.python-version` khai báo Python 3.11 / 3.13, trong khi Docker image `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` đi kèm Python 3.10.12.

**Fix:**
```toml
# pyproject.toml
requires-python = ">=3.10"
```
```
# .python-version
3.10
```

> Không nâng Python vì image không có sẵn 3.11+. Phải hạ yêu cầu xuống 3.10.

---

## 2. `uv sync` tạo `.venv` sai torch version

**Lỗi:**
```
torch 2.6.0+cu124 được cài vào .venv thay vì dùng torch 2.9.1+cu128 có sẵn
bitsandbytes / torchao lỗi AttributeError: register_constant
```

**Nguyên nhân:** `uv sync` luôn tạo `.venv` mới và resolve torch từ PyPI — tải về CUDA 12.4 thay vì dùng torch 2.9.1 có sẵn trong image.

**Fix:**
```bash
# ĐÚNG — cài thẳng vào system Python của image
uv pip install --system -e ".[docker]"
uv pip install --system unsloth

# SAI — KHÔNG dùng cái này trên Docker pod
# uv sync
```

Nếu đã lỡ chạy `uv sync`, xoá đi:
```bash
rm -rf .venv
```

---

## 3. `ModuleNotFoundError: No module named 'src'`

**Lỗi:**
```
ModuleNotFoundError: No module named 'src'
```

**Nguyên nhân:** Chạy `python3 src/app/train_sft.py` trực tiếp mà không set `PYTHONPATH`, Python không tìm thấy package `src`.

**Fix — thêm `PYTHONPATH=.` trước mọi lệnh python:**
```bash
PYTHONPATH=. python3 src/app/train_sft.py --config configs/qwen3_1b7_legal.yaml
```

Trong shell script (`run_all/sft_docker.sh`):
```bash
set -a; source .env; set +a
PYTHONPATH=. python3 src/app/train_sft.py --config configs/qwen3_1b7_legal.yaml
```

Trong test scripts, thêm `PYTHONPATH=.` vào tất cả lệnh `uv run python` / `python3`.

---

## 4. Triton lỗi không tìm thấy C compiler

**Lỗi:**
```
triton RuntimeError: Failed to find C compiler. Please specify via CC environment variable
```

**Nguyên nhân:** Docker image không có `gcc` mặc định. Unsloth dùng Triton kernel cần compile C.

**Fix:**
```bash
apt update && apt install -y gcc
```

---

## 5. Triton lỗi không tìm thấy `Python.h`

**Lỗi:**
```
fatal error: Python.h: No such file or directory
```

**Nguyên nhân:** Thiếu `python3-dev` (header files của Python cần cho Triton compilation).

**Fix:**
```bash
apt install -y python3-dev
```

> Cài cả 2 cùng lúc cho nhanh:
> ```bash
> apt update && apt install -y git tmux curl gcc python3-dev
> ```

---

## 6. Permission denied khi chạy shell script

**Lỗi:**
```
bash: ./run_all/sft_docker.sh: Permission denied
bash: ./test/test_sft.sh: Permission denied
```

**Nguyên nhân:** File `.sh` sau khi clone từ git trên Linux không có execute permission.

**Fix:**
```bash
chmod +x run_all/*.sh test/*.sh
```

---

## 7. `python` command not found

**Lỗi:**
```
bash: python: command not found
```

**Nguyên nhân:** Docker image chỉ có `python3`, không có symlink `python`.

**Fix:** Dùng `python3` thay vì `python` trong mọi lệnh:
```bash
python3 src/app/train_sft.py ...
python3 -c "..."
```

---

## 8. wandb: user is not logged in

**Lỗi:**
```
wandb: ERROR Error while calling W&B API: user is not logged in
```

**Nguyên nhân:** Biến `WANDB_API_KEY` có trong `.env` nhưng shell script không export ra environment, hoặc `wandb.login()` chưa được gọi trong code.

**Fix 1 — Export biến trong shell script:**
```bash
# Thay vì: source .env
# Dùng:
set -a; source .env; set +a
```

**Fix 2 — Login programmatically trong `train_sft.py`:**
```python
if getattr(cfg.training, "report_to", "none") == "wandb":
    import wandb
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"
```

---

## 9. wandb entity not found

**Lỗi:**
```
wandb: ERROR Error: Entity 'hoangvuhi2208-ton-duc-thang-university-org' not found
```

**Nguyên nhân:** Tên entity wandb không phải là tên org trên GitHub, và không có hậu tố `-org`.

**Fix:** Lấy đúng entity name từ wandb quickstart code (vào wandb.ai → project → copy đoạn init):
```python
wandb.init(entity="hoangvuhi2208-ton-duc-thang-university", ...)
```

Cập nhật `.env`:
```
WANDB_ENTITY=hoangvuhi2208-ton-duc-thang-university
```

> **Lưu ý:** Entity = tên team trên wandb, KHÔNG phải tên GitHub org, KHÔNG phải username cá nhân.

---

## 10. `learning_rate` TypeError

**Lỗi:**
```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

**Nguyên nhân:** OmegaConf đọc `2e-4` từ YAML dưới dạng string, không tự convert sang float.

**Fix — ép kiểu trong `src/training/args.py`:**
```python
learning_rate=float(t.learning_rate),
```

---

## 11. bitsandbytes `validate_bnb_backend_availability` error

**Lỗi:**
```
AttributeError: module 'bitsandbytes' has no attribute 'validate_bnb_backend_availability'
```

**Nguyên nhân:** `.venv` cài bitsandbytes version mới cần torch 2.7+ nhưng torch trong `.venv` là 2.6.0.

**Fix:** Xoá `.venv`, dùng `--system`:
```bash
rm -rf .venv
uv pip install --system -e ".[docker]"
uv pip install --system unsloth
```

---

## 12. Migrate từ transformers/bitsandbytes sang unsloth

**Lỗi gốc:** Training chậm (~6GB VRAM, GPU-Util thấp), cần tối ưu.

**Fix — Rewrite `src/models/base_model.py`:**
```python
from unsloth import FastLanguageModel

def load_base_model(cfg):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.base_model,
        max_seq_length=cfg.data.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.quantization.load_in_4bit if cfg.quantization.enabled else False,
        trust_remote_code=cfg.model.trust_remote_code,
        cache_dir=cfg.project.cache_dir,
    )
    return model, tokenizer
```

**Fix — Rewrite `src/models/peft_lora.py`:**
```python
from unsloth import FastLanguageModel

def apply_lora(model, cfg):
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        use_gradient_checkpointing="unsloth",  # tiết kiệm 30% VRAM
        random_state=getattr(cfg, "seed", 42),
    )
    return model
```

**Fix — Đổi `TrainingArguments` → `SFTConfig` trong `src/training/args.py`:**
```python
from trl import SFTConfig

def build_training_args(cfg, output_dir):
    return SFTConfig(
        ...,
        optim=getattr(t, "optim", "adamw_8bit"),
        max_seq_length=cfg.data.max_seq_length,
        dataset_text_field="text",
    )
```

**Kết quả:** VRAM tăng từ 6GB → 18-20GB, GPU-Util từ ~40% → 83-99%, training nhanh hơn ~2x.

---

## 13. `optim: paged_adamw_8bit` không hoạt động với unsloth

**Lỗi:**
```
ValueError: optim paged_adamw_8bit requires bitsandbytes
```

**Nguyên nhân:** Unsloth dùng optimizer riêng, không cần bitsandbytes. `paged_adamw_8bit` là của bitsandbytes.

**Fix — Đổi trong `configs/qwen3_1b7_legal.yaml`:**
```yaml
optim: adamw_8bit   # thay vì paged_adamw_8bit
```

---

## 14. `SFTTrainer` lỗi với `data_collator`

**Nguyên nhân:** `SFTConfig` (trl v1.0+) tự handle collation, truyền thêm `data_collator` gây conflict.

**Fix:** Xoá `data_collator` khỏi `SFTTrainer` constructor trong `src/training/sft_trainer.py`.

---

## 15. HuggingFace 401 Unauthorized

**Lỗi:**
```
huggingface_hub.errors.HfHubHTTPError: 401 Unauthorized
```

**Nguyên nhân:** Token HF bị revoke hoặc token có scope Read thay vì Write.

**Fix:** Tạo token mới tại [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):
- Chọn **Write** (hoặc **Fine-grained** với quyền write)
- Cập nhật `HF_TOKEN` trong `.env`

Kiểm tra token hợp lệ:
```bash
python3 -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

---

## 16. HuggingFace `RepositoryNotFoundError`

**Lỗi:**
```
huggingface_hub.errors.RepositoryNotFoundError: Repository Not Found
```

**Nguyên nhân:** Repo chưa tồn tại trên HuggingFace khi gọi `upload_folder`.

**Fix — Thêm `create_repo` trước `upload_folder`:**
```python
api.create_repo(
    repo_id=hf_repo,
    repo_type="model",
    exist_ok=True,   # không lỗi nếu đã tồn tại
    private=True,
)
api.upload_folder(...)
```

---

## 17. Eval `ValueError: Target is multiclass but average='binary'`

**Lỗi:**
```
ValueError: Target is multiclass but average='binary'. 
Please choose another average setting, one of ['macro', 'micro', 'weighted']
```

**Nguyên nhân:** Batch generation đôi khi sinh ra nhãn ngoài tập `{"relevant", "not_relevant"}` (ví dụ `"unknown"`) → thành 3 class, không dùng được `average="binary"`.

**Fix trong `scripts/eval.py`:**
```python
def _acc_f1_nli(gold, pred):
    unique = set(gold) | set(pred)
    if unique <= {"relevant", "not_relevant"}:
        f1 = f1_score(gold, pred, pos_label="relevant", average="binary", zero_division=0)
    else:
        f1 = f1_score(gold, pred, average="macro", zero_division=0)
```

---

## 18. Eval chạy chậm, không có progress bar

**Vấn đề:** Eval generate từng sample một, không biết còn bao lâu.

**Fix — Batch generation + tqdm trong `scripts/eval.py`:**
```python
from tqdm import tqdm

def generate(model, tokenizer, prompts, batch_size=8, ...):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", unit="batch"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        ...
    return results
```

Chạy eval với batch lớn:
```bash
python3 scripts/eval.py ... --batch_size 32
```

---

## 19. FutureWarning spam khi eval

**Lỗi:**
```
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated...
```

**Nguyên nhân:** unsloth dùng API cũ của torch AMP.

**Fix — Suppress ở đầu `scripts/eval.py`:**
```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

> Chỉ suppress warning, không ảnh hưởng kết quả.

---

## Tóm tắt nhanh — Checklist khi gặp lỗi

| Triệu chứng | Kiểm tra |
|---|---|
| `No module named 'src'` | Thêm `PYTHONPATH=.` trước lệnh python |
| `Permission denied` .sh | `chmod +x run_all/*.sh test/*.sh` |
| `python: not found` | Dùng `python3` thay vì `python` |
| wandb not logged in | `set -a; source .env; set +a` + `wandb.login(key=..., relogin=True)` |
| wandb entity not found | Lấy entity đúng từ wandb quickstart, không thêm `-org` |
| torch version sai | Xoá `.venv`, dùng `uv pip install --system` |
| triton C compiler | `apt install -y gcc python3-dev` |
| HF 401 Unauthorized | Tạo token Write mới, kiểm tra bằng `whoami()` |
| HF RepositoryNotFound | Thêm `create_repo(exist_ok=True)` trước `upload_folder` |
| `learning_rate` TypeError | `float(t.learning_rate)` trong args.py |
| F1 multiclass error | Check `unique <= {"relevant", "not_relevant"}` trước khi dùng binary F1 |
