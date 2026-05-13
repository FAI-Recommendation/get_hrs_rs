# 03 – Tạo Image Embedding bằng MobileNetV2

## Mục tiêu

Chuyển 2223 ảnh `.jpg` thành 2223 vector số học `.npy` (shape `(1280,)`) để notebook preprocessing đọc vào tạo `feature3` trong `items_features.csv`.

---

## Script: `get_embedding_MBNV2_optimized.py`

### Cấu hình

```python
BASE_DIR   = Path(__file__).resolve().parent / "output_10k_sample"
image_dir  = str(BASE_DIR / "images_2223_main")   # input: 2223 ảnh
output_dir = str(BASE_DIR / "embeddings_10k")     # output: 2223 .npy

TARGET_SIZE = (224, 224)   # chuẩn MobileNetV2
BATCH_SIZE  = 32
```

### Luồng xử lý

```
images_2223_main/
  abc123.jpg  (bất kỳ kích thước)
  def456.jpg
  ...
       ↓  resize → (224, 224)
       ↓  mobilenet_v2.preprocess_input()   ← chuẩn hóa đúng cách
       ↓  MobileNetV2(include_top=False, pooling='avg')
       ↓  model.predict(batch_32_ảnh)
       ↓  shape (32, 1280) → lưu từng file
embeddings_10k/
  abc123.npy  shape (1280,)
  def456.npy  shape (1280,)
  ...
```

---

## Tại sao dùng `preprocess_input()` thay vì `/ 255.0`?

| Cách | Công thức | Vấn đề |
|---|---|---|
| `img / 255.0` ❌ | Normalize về [0, 1] | Sai — MobileNetV2 được train với [-1, 1] |
| `mobilenet_v2.preprocess_input()` ✅ | `(img / 127.5) - 1` → [-1, 1] | Đúng với pretrained weights |

Dùng sai preprocessing → embedding sai → similarity matrix sai → hit_ratio thấp.

---

## Tại sao batch 32 thay vì từng ảnh?

| Cách | Tốc độ ước tính (2223 ảnh, RTX 4050) |
|---|---|
| Từng ảnh một (version cũ) | ~3-4 phút |
| Batch 32 (version hiện tại) | ~15-30 giây |

---

## GPU Memory Growth

```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

Bắt buộc với RTX 4050 (6GB VRAM). Không có dòng này TensorFlow chiếm hết VRAM ngay lúc khởi động → có thể crash.

---

## Output

```
embeddings_10k/
  abc123.npy   → numpy array shape (1280,), dtype float32
  def456.npy
  ...
  (2223 file tổng cộng)
```

Tên file `.npy` = tên file `.jpg` bỏ đuôi mở rộng.

---

## Chạy script

```powershell
# Bước 1: copy 2223 ảnh vào images_2223_main/
.venv\Scripts\python.exe rs\get10k_data\copy_images_10k.py

# Bước 2: tạo embedding
.venv\Scripts\python.exe rs\get10k_data\get_embedding_MBNV2_optimized.py

# Monitor GPU trong khi chạy
nvidia-smi -l 1
```
