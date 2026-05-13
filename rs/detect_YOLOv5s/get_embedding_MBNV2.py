import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Đường dẫn tới thư mục
image_dir = 'D:/HK1_2024-2025___NOW/DA_CNTT/New_RS/VibrentDataset/images_cropped'
output_dir = 'D:/HK1_2024-2025___NOW/DA_CNTT/New_RS/embeddings/embeddings_MBNV2_full'

# Khởi tạo mô hình MobileNetV2
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def get_image_embedding(image_path):
    """Trích xuất vector embedding từ hình ảnh bằng MobileNetV2."""
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    embedding = model.predict(img_array)
    return embedding

# Lặp qua tất cả hình ảnh trong thư mục
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        embedding = get_image_embedding(image_path)
        # Lưu vector embedding vào file .npy với tên tương ứng
        np.save(os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}.npy'), embedding)

print("Đã hoàn tất việc trích xuất và lưu vector embedding!")