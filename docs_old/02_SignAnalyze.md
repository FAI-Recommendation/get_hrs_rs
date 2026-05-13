# Giải Nghĩa Cấu Trúc Đặt Tên Tập Dữ Liệu

Hệ thống sử dụng các bộ dữ liệu khác nhau kết hợp với các kỹ thuật tiền xử lý (Preprocessing) đã được cấu trúc hóa thông qua các tên thư mục con trong dữ liệu gốc. Tài liệu này cung cấp định nghĩa tra cứu nhanh.

## 1. Nguồn Gốc Dữ Liệu (Core Datasets)
*   **`vcr`**: Viết tắt của **Vibrent Clothes Rental Dataset** (Tập dữ liệu thuê quần áo - Nguồn từ Kaggle). Đây là tệp dữ liệu trọng tâm đang được ứng dụng.
*   **`rtr`**: Khả năng là **Rent The Runway** (Tập dữ liệu hệ thống thời trang lớn khác đối chứng kết quả).

## 2. Các Chỉ Số Thuộc Tính Rút Gọn (Sampling Percentages)
Khi làm việc với Big Data trên GCN, hệ thống phân bổ các phiên bản mẫu thu nhỏ (Sampling):
*   **Ký hiệu `p`**: Nghĩa là Percentage (Phần trăm).
*   **Ví dụ `vcr1p`, `vcr5p`**: Quần thể dữ liệu đã được lấy mẫu ở tỷ lệ `1%` hoặc `5%` đối với tương tác mua sắm gốc, hoặc lọc người dùng có lượng tương tác tối thiểu xác định nhằm phục vụ quá trình huấn luyện nhanh (Fast Training).

## 3. Các Phương Pháp Thực Thi Tiền Xử Lý (Processing Techniques)
Các kỹ thuật làm đa dạng cách phân giải thông tin từ hình ảnh/văn bản trước khi đưa vào GCN:

*   **`vcr_detect`**: Dữ liệu ảnh đầu vào được chạy qua kiến trúc **YOLO (Object Detection)** để tiến hành bóc tách phông nền, giữ lại vector đặc trưng vùng áo quần lõi nhất.
*   **`img_m1`, `img_m2`, `img_m3`**: Ký hiệu các **chiến lược gộp embedding ảnh** dùng trong pipeline tạo `items_features.csv` và các biến thể benchmark ở `Eval_All.ipynb`:
    *   **M1**: `mean` - lấy trung bình các embedding ảnh.
    *   **M2**: `weighted` - gộp theo trọng số.
    *   **M3**: `max` - lấy giá trị lớn nhất theo từng chiều.
*   **`aggregation` (Kỹ thuật Tổng hợp)**: Nhánh chiến thuật gộp chung các vector đặc trưng lại với nhau trước khi tính similarity.
*   **`late_fusion` (Kỹ thuật Gộp Nối Tiếp Muộn)**: Xử lý hai tín hiệu Hình ảnh và Văn bản song song độc lập, sau đó hợp nhất ở bước cuối để tạo similarity matrix.
*   **`multiheadattention` & `weightattention`**: Các biến thể gộp embedding dùng attention để học trọng số thay vì gộp cứng theo mean/max.
*   **`pca` (Principal Component Analysis)**: Kỹ thuật giảm chiều không gian vector trước khi tính similarity, nhằm nén dữ liệu và thử nghiệm ảnh hưởng của giảm chiều.

---

**Khuyến nghị sử dụng cấu trúc cho thử nghiệm thực tế:**
1.  **Kiểm thử và vận hành nhanh (Fast Testing):** Tập trung chạy các thư mục có hậu tố **`vcr1p`** (Quy mô dữ liệu gọn nhẹ, tránh quá tải Out Of Memory).
2.  **Triển khai theo mức độ mô hình mạnh nhất (Deployment level):** Môi trường thử nghiệm có hệ thống tiền xử lý **`late_fusion`** hoặc **`multiheadattention`** là những phiên bản chuyên sâu và ưu việt ở tính chuẩn xác cao.