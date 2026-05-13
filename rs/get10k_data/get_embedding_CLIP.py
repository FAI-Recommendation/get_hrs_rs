"""
Script trích embedding ảnh bằng CLIP (ViT-B/32) với batch processing & GPU.

INPUT : output_10k_sample/images_2223_main/      (2223 ảnh .jpg)
OUTPUT: output_10k_sample/embeddings_clip_10k/   (2223 file .npy, mỗi file shape (512,))

Model: openai/clip-vit-base-patch32
  - ViT-B  = Vision Transformer Base
  - /32    = chia ảnh thành các patch 32×32 pixel
  - Output: vector 512 chiều (thay vì 1280 của MobileNetV2)

Tại sao CLIP tốt hơn MobileNetV2 cho fashion?
  - MobileNetV2 được train trên ImageNet (phân loại 1000 class chung)
    → vector 1280-D encode "vật thể là gì"
  - CLIP được train trên 400 triệu cặp (ảnh, caption) từ internet
    → vector 512-D encode "ảnh trông như thế nào + ngữ nghĩa"
    → similarity trong không gian CLIP phản ánh "outfit này có vẻ giống nhau không"
    → không cần PCA vì 512-D tương đương BERT 768-D về semantic
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ============ CẤU HÌNH ============
BASE_DIR   = Path(__file__).resolve().parent / "output_10k_sample"
image_dir  = str(BASE_DIR / "images_2223_main")
output_dir = str(BASE_DIR / "embeddings_clip_10k")

BATCH_SIZE = 32

# ============ SETUP GPU ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'✅ GPU' if device.type == 'cuda' else '⚠️  CPU'}: {device}")

os.makedirs(output_dir, exist_ok=True)

# ============ LOAD MODEL ============
# openai/clip-vit-base-patch32:
#   - "vit-base"   : Vision Transformer cỡ Base (86M params)
#   - "patch32"    : ảnh 224×224 được chia thành (224/32)² = 49 patch
#   - Được Hugging Face host lại từ weights gốc của OpenAI
#   - Tại sao không dùng clip-vit-large? → large (307M) nặng gấp 4x,
#     với 2223 ảnh thì base đã đủ tốt và nhanh hơn nhiều
MODEL_NAME = "openai/clip-vit-base-patch32"
print(f"Loading {MODEL_NAME} ...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model     = CLIPModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print("✅ Model loaded")


# ============ HÀM XỬ LÝ BATCH ============
def load_images_batch(image_paths, valid_files):
    """
    Load batch ảnh bằng PIL.
    Nếu 1 ảnh lỗi → bỏ ảnh đó, KHÔNG bỏ cả batch.
    Trả về (list[PIL.Image], good_files).
    """
    images     = []
    good_files = []
    for path, fname in zip(image_paths, valid_files):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            good_files.append(fname)
        except Exception as e:
            print(f"❌ Lỗi đọc {path}: {e}")
    return images, good_files


def get_clip_embeddings(images: list) -> np.ndarray:
    """
    Trả về array (N, 512) — CLIP image embeddings đã L2-normalize.
    """
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # Dùng vision_model → pooler_output → visual_projection → 512-D tensor
        vision_output = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(vision_output.pooler_output)  # (N, 512) tensor
        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


# ============ MAIN LOOP ============
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
print(f"Tìm thấy {len(image_files)} ảnh trong {image_dir}")

processed = 0

for batch_idx in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing batches"):
    batch_files = image_files[batch_idx : batch_idx + BATCH_SIZE]
    batch_paths = [os.path.join(image_dir, f) for f in batch_files]

    images, good_files = load_images_batch(batch_paths, batch_files)
    if not images:
        continue

    embeddings = get_clip_embeddings(images)   # (N, 512)

    for filename, embedding in zip(good_files, embeddings):
        out_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")
        np.save(out_path, embedding)
        processed += 1

print(f"\n✅ Hoàn tất! Đã lưu {processed}/{len(image_files)} embeddings tại {output_dir}")
print(f"   Mỗi embedding có kích thước: (512,)")
print(f"\n💡 Để dùng trong pipeline: thay embeddings_10k → embeddings_clip_10k")
print(f"   Và bỏ bước PCA vì 512-D không cần giảm chiều về 768.")
