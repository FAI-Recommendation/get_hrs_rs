# 04 – Chiến Lược Chọn Ảnh Chính (`displayOrder == 0`)

## Vấn đề

Mỗi outfit trong VCR dataset có **4-5 ảnh** với các góc độ khác nhau. Cần chọn đúng 1 ảnh đại diện để tạo embedding.

## Ý nghĩa `displayOrder`

| displayOrder | Nội dung ảnh | Thông tin chứa |
|---|---|---|
| **0** | Model mặc toàn thân — góc trực diện | Form dáng, màu sắc, phong cách tổng thể ✅ |
| 1 | Chụp từ phía sau | Thiếu thông tin phía trước |
| 2 | Cận cảnh chất liệu vải | Chỉ có texture, không có form |
| 3 | Ảnh mác / phụ kiện | Không đại diện cho outfit |

---

## Tác động đến embedding

```
Ảnh 0 (toàn thân đỏ, sang trọng)
  → MobileNetV2 → vector: "váy dài, màu đỏ, formal"
  → Cosine similarity cao với: váy dạ hội khác ✅

Ảnh 2 (cận cảnh vải ren)
  → MobileNetV2 → vector: "texture ren trắng mịn"
  → Cosine similarity cao với: áo thun có vải ren ❌ (sai hoàn toàn)
```

Chọn sai ảnh → similarity matrix sai → LightGCN gợi ý sai → hit_ratio thấp.

---

## Cách áp dụng trong code

### Trong `slipt_10k_sample.py`

```python
def pick_main_picture_rows(pictures):
    main_pictures = pictures[pictures["displayOrder"] == 0].copy()
    main_pictures = main_pictures.drop_duplicates(subset=["outfit.id"], keep="first")
    return main_pictures
    # Kết quả: 1 outfit = 1 ảnh chính duy nhất
```

### Trong notebook preprocessing

```python
pictures = pictures[pictures['displayOrder'] == 0]
```

---

## Cơ sở học thuật

| Bài báo | Liên quan |
|---|---|
| **DeepFashion2** (Ge et al., CVPR 2019) | Định nghĩa "Frontal View" là ảnh chứa nhiều Global Features nhất |
| **VBPR** (He & McAuley, AAAI 2016) | Chứng minh dùng Primary Image giúp embedding hội tụ nhanh và chính xác hơn |

### Citation cho báo cáo

> *"Dựa trên phương pháp của Ge et al. (2019) trong DeepFashion2, nghiên cứu này sử dụng ảnh có `displayOrder = 0` (Canonical View) làm đại diện cho mỗi outfit, giúp vector embedding chứa Global Features và giảm Visual Noise so với các góc chụp cận cảnh."*
