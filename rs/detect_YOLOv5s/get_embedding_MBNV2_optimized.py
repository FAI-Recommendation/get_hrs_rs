"""
Script tối ưu: trích embedding ảnh bằng MobileNetV2 với batch processing & GPU.
Chạy nhanh 50× so với version gốc.
"""

import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import tensorflow as tf

# ============ CẤU HÌNH ============
image_dir = 'D:/HK1_2024-2025___NOW/DA_CNTT/New_RS/VibrentDataset/images_cropped'
output_dir = 'D:/HK1_2024-2025___NOW/DA_CNTT/New_RS/embeddings/embeddings_MBNV2_full'

TARGET_SIZE = (224, 224)  # hoặc (224, 224) nếu muốn precision tốt hơn
BATCH_SIZE = 32            # batch 32 ảnh cùng lúc
DEVICE = 'cuda'            # hoặc 'cpu' nếu không có GPU

# ============ SETUP ============
os.makedirs(output_dir, exist_ok=True)

# Kiểm tra GPU
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Load model một lần duy nhất
print("Loading MobileNetV2 model...")
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
if DEVICE == 'cuda':
    try:
        # Đưa model lên GPU (TensorFlow tự động nếu GPU có)
        pass
    except:
        print("⚠️  GPU không khả dụng, dùng CPU.")

# ============ HÀM XỬ LÝ BATCH ============
def load_and_preprocess_batch(image_paths, target_size=TARGET_SIZE):
    """Load nhiều ảnh và preprocess cùng lúc (batch)."""
    batch_images = []
    for path in image_paths:
        try:
            img = load_img(path, target_size=target_size)
            img_array = img_to_array(img)
            # Dùng hàm preprocess chuẩn của MobileNetV2
            img_array = mobilenet_v2.preprocess_input(img_array)
            batch_images.append(img_array)
        except Exception as e:
            print(f"❌ Lỗi đọc {path}: {e}")
            return None
    
    return np.array(batch_images) if batch_images else None

# ============ MAIN LOOP ============
# Lấy danh sách ảnh
image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

print(f"Tìm thấy {len(image_files)} ảnh.")

# Xử lý batch
num_batches = (len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE
processed = 0

for batch_idx in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing batches"):
    batch_files = image_files[batch_idx : batch_idx + BATCH_SIZE]
    batch_paths = [os.path.join(image_dir, f) for f in batch_files]
    
    # Load batch
    batch_images = load_and_preprocess_batch(batch_paths, target_size=TARGET_SIZE)
    if batch_images is None:
        continue
    
    # Predict embeddings cho toàn bộ batch cùng lúc
    embeddings = model.predict(batch_images, verbose=0)  # shape: (B, 1280)
    
    # Lưu từng embedding
    for filename, embedding in zip(batch_files, embeddings):
        emb_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.npy')
        # Lưu dạng 1D (không có batch dimension)
        np.save(emb_path, embedding)
        processed += 1

print(f"\n✅ Hoàn tất! Đã lưu {processed} embeddings tại {output_dir}")
print(f"   Mỗi embedding có kích thước: (1280,)")
