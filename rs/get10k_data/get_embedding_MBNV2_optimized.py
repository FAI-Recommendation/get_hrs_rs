"""
Script trích embedding ảnh bằng MobileNetV2 với batch processing & GPU.

INPUT : output_10k_sample/images_2223_main/   (2223 ảnh .jpg)
OUTPUT: output_10k_sample/embeddings_10k/     (2223 file .npy, mỗi file shape (1280,))
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

# ============ CẤU HÌNH ============
BASE_DIR   = Path(__file__).resolve().parent / "output_10k_sample"
image_dir  = str(BASE_DIR / "images_2223_main")
output_dir = str(BASE_DIR / "embeddings_10k")

TARGET_SIZE = (224, 224)
BATCH_SIZE  = 32

# ============ SETUP GPU ============
# Bật memory growth để tránh TensorFlow chiếm hết 6GB VRAM của RTX 4050
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU: {[g.name for g in gpus]}")
else:
    print("⚠️  Không tìm thấy GPU, chạy bằng CPU.")

os.makedirs(output_dir, exist_ok=True)

# ============ LOAD MODEL ============
print("Loading MobileNetV2 model...")
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')


# ============ HÀM XỬ LÝ BATCH ============
def load_and_preprocess_batch(image_paths, valid_files, target_size=TARGET_SIZE):
    """
    Load và preprocess batch ảnh.
    Nếu 1 ảnh lỗi → bỏ ảnh đó, KHÔNG bỏ cả batch.
    Trả về (batch_array, good_files) — good_files là tên ảnh load thành công.
    """
    batch_images = []
    good_files   = []
    for path, fname in zip(image_paths, valid_files):
        try:
            img       = load_img(path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = mobilenet_v2.preprocess_input(img_array)
            batch_images.append(img_array)
            good_files.append(fname)
        except Exception as e:
            print(f"❌ Lỗi đọc {path}: {e}")

    if not batch_images:
        return None, []
    return np.array(batch_images), good_files


# ============ MAIN LOOP ============
image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
print(f"Tìm thấy {len(image_files)} ảnh trong {image_dir}")

processed = 0

for batch_idx in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing batches"):
    batch_files = image_files[batch_idx : batch_idx + BATCH_SIZE]
    batch_paths = [os.path.join(image_dir, f) for f in batch_files]

    batch_array, good_files = load_and_preprocess_batch(batch_paths, batch_files)
    if batch_array is None:
        continue

    embeddings = model.predict(batch_array, verbose=0)  # shape: (B, 1280)

    for filename, embedding in zip(good_files, embeddings):
        emb_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.npy')
        np.save(emb_path, embedding)
        processed += 1

print(f"\n✅ Hoàn tất! Đã lưu {processed}/{len(image_files)} embeddings tại {output_dir}")
print(f"   Mỗi embedding có kích thước: (1280,)")
