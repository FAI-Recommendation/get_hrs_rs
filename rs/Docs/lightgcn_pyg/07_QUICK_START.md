# Quick Start — Setup & Run CombiGCN (PyG)

Mục đích: hướng dẫn nhanh các bước cần làm để chạy pipeline `rs/lightgcn_pyg` trên máy local hoặc remote GPU pod.

**Tóm tắt các bước**
- Clone repo → chỉnh `.env` → chạy `setup_env.sh` → chạy smoke test → train.

**1) Clone repo**

```bash
git clone https://github.com/<org>/<repo>.git  # hoặc dùng PAT nếu private
cd <repo>
```

**2) Tạo và chỉnh `.env`**

```bash
cp .env.example .env
# sửa các biến: DATA_DIR, OUTPUT_DIR, HF_TOKEN, MULTIMODAL_METHOD, MULTIMODAL_PCA_COMPONENTS nếu cần
nano .env
```

Lưu ý: các script `rs/lightgcn_pyg/scripts/*` sẽ tự động `source .env` nếu file tồn tại.

**3) Chuẩn bị môi trường (system + PyG wheels)**

- Đảm bảo có `sudo` trên pod hoặc chạy như root.
- Đánh dấu file setup executable và chạy:

```bash
chmod +x rs/lightgcn_pyg/scripts/setup_env.sh
# chạy với index wheel mặc định (script mặc định dùng torch-2.9.0+cu128)
bash rs/lightgcn_pyg/scripts/setup_env.sh

# nếu cần override wheel index (ví dụ khác CUDA), đặt TORCH_WHL_URL:
TORCH_WHL_URL=https://data.pyg.org/whl/torch-2.1.2+cu121.html bash rs/lightgcn_pyg/scripts/setup_env.sh
```

Ghi chú:
- `setup_env.sh` cài `git tmux curl build-essential python3-dev cmake libomp-dev`, sau đó cài `uv` và các package Python cần thiết.
- Nếu bạn đang dùng Docker image có sẵn PyTorch (vd. `pytorch/pytorch:2.1.2-cuda12.1`), **không** cài lại `torch` — chỉ cài companion wheels bằng `TORCH_WHL_URL` phù hợp.

**4) Chạy smoke tests (nhanh, 20 epochs)**

```bash
bash rs/lightgcn_pyg/scripts/test/test_lightgcn.sh    # LightGCN thuần
bash rs/lightgcn_pyg/scripts/test/test_img_only.sh   # CombiGCN + img_only
bash rs/lightgcn_pyg/scripts/test/test_multimodal.sh # CombiGCN + multimodal
bash rs/lightgcn_pyg/scripts/test/test_tfidf.sh      # CombiGCN + tfidf
```

Các script này đã source `.env` và chạy `python train.py` với tham số test ngắn (epoch=20, save_flag=0).

**5) Chạy training dài (ví dụ một variant)**

```bash
bash rs/lightgcn_pyg/scripts/run_all/run_img_only.sh
```

Hoặc chạy đầy đủ 4 variants:

```bash
bash rs/lightgcn_pyg/scripts/run_all/run_all.sh
```

Chạy trong `tmux` khi training lâu:

```bash
tmux new -s train
# trong session tmux
bash rs/lightgcn_pyg/scripts/run_all/run_img_only.sh
# detach: Ctrl+B rồi D
tmux attach -t train
```

**6) Cấu hình multimodal (PCA / fusion)**

- Biến config trong `.env`:
  - `MULTIMODAL_METHOD` = `late_fusion` | `aggregation` | `pca` | `attention` (mặc định `late_fusion`)
  - `MULTIMODAL_PCA_COMPONENTS` = integer (nếu để trống và `method=pca`, code sẽ tự chọn: `min(text_dim + img_dim, 768)`)

Ví dụ trong `.env`:

```
MULTIMODAL_METHOD=late_fusion
MULTIMODAL_PCA_COMPONENTS=256
```

**7) Wheel index (PyG companion wheels)**

- Nếu cần cài `pyg_lib`, `torch_scatter`, `torch_sparse`, set `TORCH_WHL_URL` trước khi chạy `setup_env.sh`.

**8) Kiểm tra sau khi cài**

```bash
python3 -c "import torch, torch_geometric, torch_sparse; print(torch.__version__, torch.cuda.is_available())"
```

File logs & weights:
- TensorBoard: path in `train.py` -> `tensorboard/<sim_type>/...`
- Weights: `rs/lightgcn_pyg/weights/<sim_type>/...`
- Results: `rs/lightgcn_pyg/output/<sim_type>/combigcn.result`

Nếu bạn muốn mình thêm: bàn luận về chọn `batch_size` / `embed_size` theo VRAM, hoặc thêm CLI flags cho `MULTIMODAL_*`, báo mình biết.
