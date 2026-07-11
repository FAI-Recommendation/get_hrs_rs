# Hướng Dẫn Cấu Hình (Config) & Chạy Huấn Luyện (Training)

Tài liệu này hướng dẫn chi tiết cách thiết lập các file cấu hình, định nghĩa siêu tham số (Hyperparameters), và cú pháp thực thi các lệnh chạy thử nghiệm (smoke test) cũng như chạy huấn luyện chính thức (full run) cho cả 3 mô hình: **CombiGCN**, **BM3**, và **FREEDOM**.

---

## 1. Các File Tài Liệu Tham Chiếu Gốc (Source Docs)
Các cấu hình và lệnh chạy này được chiết xuất từ:
*   [04_training_pipeline.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/models_pyg_training_latest/04_training_pipeline.md) — Chi tiết vòng lặp huấn luyện, hàm Loss BPR và các siêu tham số.
*   [06_scripts_experiments.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/models_pyg_training_latest/06_scripts_experiments.md) — Cấu trúc hệ thống script tự động chạy 24 thực nghiệm.
*   [07_remote_gpu.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/models_pyg_training_latest/07_remote_gpu.md) — Hướng dẫn cài đặt môi trường trên remote GPU (RunPod/Vast.ai) và lệnh khởi chạy.

---

## 2. Cấu Hình Môi Trường (`.env`)
Trước khi chạy huấn luyện, cần khai báo các biến môi trường cơ bản trong file `.env` đặt tại thư mục root của project:

```ini
# Đường dẫn chứa thư mục dữ liệu đầu vào (Ví dụ: clip_10k_sample hoặc mbnv2_10k_sample)
DATA_DIR=./get10k_data

# Thư mục lưu kết quả log huấn luyện (.result, tensorboard, checkpoint)
OUTPUT_DIR=./rs/lightgcn_pyg/output

# Thiết lập GPU chạy
CUDA_VISIBLE_DEVICES=0
GPU_ID=0

# Phương pháp fuse đặc trưng đa phương thức (late_fusion / aggregation / pca / attention)
MULTIMODAL_METHOD=late_fusion
MULTIMODAL_PCA_COMPONENTS=256

# Tài khoản giám sát (Optional)
WANDB_ENTITY=your-wandb-username
HF_REPO_ID=your-hf-repository
```

---

## 3. Các Siêu Tham Số Huấn Luyện Chung (Command Arguments)

Khi khởi chạy file huấn luyện `train.py`, bạn có thể tùy biến các tham số sau thông qua Command Line:

| Tham Số Dòng Lệnh | Giá Trị Smoke Test | Giá Trị Full Train | Ý Nghĩa |
| :--- | :--- | :--- | :--- |
| `--data_path` | Tùy chọn | Tùy chọn | Đường dẫn chứa dataset đã xử lý (ví dụ: `../../get10k_data/clip_10k_sample`) |
| `--model` | `combigcn`/`bm3`/`freedom` | `combigcn`/`bm3`/`freedom` | Chọn mô hình huấn luyện |
| `--sim_type` | `none`/`img_only`/`tfidf`/`multimodal`/`multimodal_attention` | (Như cột trái) | Cách xây dựng ma trận tương đồng hoặc cơ chế fusion đa phương thức |
| `--epoch` | `20` | `1000` | Số lượng Epoch chạy huấn luyện tối đa |
| `--batch_size` | `1024` | `8192` | Kích thước batch huấn luyện (điều chỉnh theo bộ nhớ VRAM của GPU) |
| `--lr` | `0.001` | `0.001` | Tốc độ học của Optimizer Adam |
| `--regs` | `[1e-5]` | `[1e-5]` hoặc `[1e-4]` | Trọng số phạt L2 Regularization chống quá khớp (overfitting) |
| `--embed_size` | `64` | `512` | Kích thước vector nhúng ID (User/Item) |
| `--layer_size` | `"[64,64,64]"` | `"[512,512,512,512]"` | Cấu trúc số tầng lan truyền của đồ thị kề |
| `--eval_interval` | `10` | `40` | Tần suất đánh giá mô hình trên tập Test (sau mỗi N Epoch) |
| `--early_stop_steps`| `3` | `0` (hoặc `5`) | Số chu kỳ đánh giá không cải thiện để dừng sớm (0 để tắt) |
| `--save_flag` | `0` (không lưu) | `1` (có lưu) | Tùy chọn lưu lại checkpoint mô hình tốt nhất (`.pt` / `.pth`) |
| `--gpu_id` | `0` | `0` | Chỉ định GPU thực thi (`0`, `1`, hoặc `-1` để chạy CPU) |

---

## 4. Hướng Dẫn Chạy Huấn Luyện Cho Từng Model

### 4.1. Mô hình CombiGCN (Dual-Graph GCN)
CombiGCN có chế độ chạy không dùng modal (CF thuần - tương tự LightGCN) khi thiết lập `--sim_type none`.

*   **Lệnh Chạy Kiểm Thử Nhanh (Smoke Test - 20 epochs):**
    ```bash
    python3 train.py \
      --model combigcn \
      --data_path ../../get10k_data/clip_10k_sample \
      --sim_type img_only \
      --epoch 20 \
      --eval_interval 10 \
      --early_stop_steps 3 \
      --save_flag 0 \
      --embed_size 64 \
      --layer_size "[64,64,64]" \
      --gpu_id 0
    ```
*   **Lệnh Chạy Đầy Đủ (Full Training - 1000 epochs):**
    ```bash
    python3 train.py \
      --model combigcn \
      --data_path ../../get10k_data/clip_10k_sample \
      --sim_type multimodal \
      --epoch 1000 \
      --eval_interval 40 \
      --early_stop_steps 5 \
      --save_flag 1 \
      --embed_size 512 \
      --layer_size "[512,512,512,512]" \
      --gpu_id 0
    ```

---

### 4.2. Mô hình BM3 (Bootstrap Latent Representations)
BM3 bổ sung thêm thành phần hàm lỗi Tương Phản tự giám sát (Bootstrap Contrastive Loss) giữa các modality.

*   **Các tham số bổ sung đặc thù:**
    *   `--cl_weight`: Trọng số của Contrastive Loss (Mặc định thường chọn từ `0.1` đến `1.0`).
*   **Lệnh Chạy Kiểm Thử Nhanh (Smoke Test - 20 epochs):**
    ```bash
    python3 train.py \
      --model bm3 \
      --data_path ../../get10k_data/clip_10k_sample \
      --sim_type multimodal \
      --epoch 20 \
      --cl_weight 0.1 \
      --eval_interval 10 \
      --early_stop_steps 3 \
      --save_flag 0 \
      --embed_size 64 \
      --layer_size "[64,64,64]" \
      --gpu_id 0
    ```
*   **Lệnh Chạy Đầy Đủ (Full Training - 1000 epochs):**
    ```bash
    python3 train.py \
      --model bm3 \
      --data_path ../../get10k_data/clip_10k_sample \
      --sim_type multimodal_attention \
      --epoch 1000 \
      --cl_weight 0.5 \
      --eval_interval 40 \
      --early_stop_steps 5 \
      --save_flag 1 \
      --embed_size 512 \
      --layer_size "[512,512,512,512]" \
      --gpu_id 0
    ```

---

### 4.3. Mô hình FREEDOM (Frozen & Denoising Graphs)
FREEDOM dựng đồ thị $K$-lân cận tĩnh cho Item từ đặc trưng modal và áp dụng InfoNCE loss để căn chỉnh không gian biểu diễn giữa luồng Collaborative Filtering và đồ thị đặc trưng.

*   **Các tham số bổ sung đặc thù:**
    *   `--cl_weight`: Trọng số của InfoNCE contrastive loss.
    *   `--knn_k`: Số lượng lân cận dùng để dựng frozen item-item kNN graph (ví dụ: `10` hoặc `20`).
*   **Lệnh Chạy Kiểm Thử Nhanh (Smoke Test - 20 epochs):**
    ```bash
    python3 train.py \
      --model freedom \
      --data_path ../../get10k_data/clip_10k_sample \
      --sim_type img_only \
      --epoch 20 \
      --cl_weight 0.1 \
      --knn_k 10 \
      --eval_interval 10 \
      --early_stop_steps 3 \
      --save_flag 0 \
      --embed_size 64 \
      --layer_size "[64,64,64]" \
      --gpu_id 0
    ```
*   **Lệnh Chạy Đầy Đủ (Full Training - 1000 epochs):**
    ```bash
    python3 train.py \
      --model freedom \
      --data_path ../../get10k_data/clip_10k_sample \
      --sim_type multimodal \
      --epoch 1000 \
      --cl_weight 0.2 \
      --knn_k 20 \
      --eval_interval 40 \
      --early_stop_steps 5 \
      --save_flag 1 \
      --embed_size 512 \
      --layer_size "[512,512,512,512]" \
      --gpu_id 0
    ```

---

## 5. Sử Dụng Các Script Chạy Thực Nghiệm Tự Động

Thư mục `rs/lightgcn_pyg/scripts/` chứa các cấu trúc script được viết sẵn để tự động kích hoạt tuần tự các cấu hình thử nghiệm tương ứng:

```
rs/lightgcn_pyg/scripts/
├── 01_run_all_clip/       # CombiGCN + CLIP, chạy full 1000 epochs
├── 01_test_clip/          # CombiGCN + CLIP, chạy test nhanh 20 epochs
├── 02_run_all_mbnv2/      # CombiGCN + MobileNetV2, chạy full
├── 03_run_bm3_clip/       # BM3 + CLIP, chạy full
├── 03_test_bm3_clip/      # BM3 + CLIP, chạy test nhanh
├── 04_run_bm3_mbnv2/      # BM3 + MobileNetV2, chạy full
├── 05_run_freedom_clip/   # FREEDOM + CLIP, chạy full
└── 06_run_freedom_mbnv2/  # FREEDOM + MobileNetV2, chạy full
```

### Cách thực thi:
Di chuyển vào thư mục code chính và cấp quyền thực thi cho các file `.sh`:
```bash
cd rs/lightgcn_pyg
chmod +x scripts/**/*.sh
```

*   **Chạy kiểm thử toàn bộ các biến thể của mô hình trong 1 group:**
    ```bash
    bash scripts/01_test_clip/test_all.sh
    ```
*   **Chạy huấn luyện đầy đủ (1000 epochs) cho toàn bộ variants của BM3 với CLIP:**
    ```bash
    bash scripts/03_run_bm3_clip/run_all.sh
    ```
*   **Chạy riêng lẻ một biến thể nhất định (ví dụ: chỉ chạy chế độ Image Only):**
    ```bash
    bash scripts/03_run_bm3_clip/run_img_only.sh
    ```

---

## 6. Giám Sát Tiến Trình Huấn Luyện (Training Monitoring)

### 6.1. Sử dụng TensorBoard (SSH Tunneling nếu chạy máy chủ remote)
Để xem biểu đồ độ giảm loss, độ tăng Recall@K, Precision@K, NDCG@K trên máy local khi đang train trên remote GPU:

1.  **Chạy TensorBoard trên server (qua terminal SSH):**
    ```bash
    tensorboard --logdir rs/lightgcn_pyg/tensorboard/ --host 0.0.0.0 --port 6006
    ```
2.  **Mở SSH Tunnel trên terminal máy local của bạn:**
    ```bash
    ssh -L 6006:localhost:6006 root@<pod-ip> -p <port-ssh>
    ```
3.  Truy cập link sau trên trình duyệt Web: `http://localhost:6006`

### 6.2. Đồng bộ hóa với Weights & Biases (Wandb)
Để tự động ghi nhận log và so sánh chéo 24 thực nghiệm trực tiếp trên web Cloud mà không cần cấu hình SSH Port Forwarding:
1.  Đăng nhập trên máy chủ: `wandb login` và điền mã API Key của bạn.
2.  Thêm cờ `--use_wandb 1` và `--wandb_project <project_name>` vào lệnh chạy huấn luyện.

---

## 7. Khôi Phục & Sử Dụng Model Đã Huấn Luyện (Load Checkpoint)
Sau khi kết thúc quá trình huấn luyện, file trọng số tốt nhất được lưu lại tại thư mục `weights/`. Đoạn mã Python dưới đây minh họa cách load lại model để chạy dự đoán/đánh giá (inference):

```python
import torch
from model import CombiGCN  # Hoặc BM3, FREEDOM tương ứng

# 1. Load file checkpoint
checkpoint_path = 'weights/img_only/layers_3_dim_64/lr_0.001_reg_1e-05/best_model.pt'
checkpoint = torch.load(checkpoint_path)

# 2. Khởi tạo lại kiến trúc model từ tham số lưu trong checkpoint
model = CombiGCN(
    n_users=checkpoint['args']['n_users'],
    n_items=checkpoint['args']['n_items'],
    embed_dim=checkpoint['args']['embed_size'],
    n_layers=len(checkpoint['args']['layer_size']),
    decay=checkpoint['args']['regs'][0],
)

# 3. Nạp trọng số và chuyển sang chế độ đánh giá (evaluation mode)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Nạp model thành công từ Epoch: {checkpoint['epoch']}")
```
