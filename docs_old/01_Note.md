# Tổng Quan Hệ Thống Gợi Ý Thời Trang

Dự án Recommendation System được chia làm 2 mô-đun cốt lõi, phục vụ cho hai mục đích và phương pháp học máy (Machine Learning) khác nhau.

## 1. Mô-đun `rs_img_app`: Khuyến nghị Dựa Trên Hình Ảnh (Visual Similarity)
*   **Mục tiêu:** Tìm các sản phẩm có ngoại hình giống nhau (màu sắc, kiểu dáng, hoa văn) dựa trên một hình ảnh đầu vào.
*   **Dữ liệu sử dụng:** Chỉ sử dụng dữ liệu **Hình ảnh** (Pixel ảnh).
*   **Kiến trúc Thuật toán:**
    *   **YOLOv5s:** Phát hiện (Object Detection) và cắt chuẩn xác vùng chứa trang phục/quần áo trong ảnh gốc, loại bỏ phông nền dư thừa.
    *   **MobileNetV2:** Trích xuất đặc trưng ảnh (Feature Extraction) để chuyển đổi ảnh thành các vector nhúng (Embeddings).
    *   **FAISS:** Công cụ tìm kiếm vector tương đồng cục bộ với tốc độ cao.
*   **Ứng dụng thực tế:** Tính năng "Tìm sản phẩm tương tự" hoặc "Tìm kiếm bằng hình ảnh".

## 2. Mô-đun `hrs_system`: Hệ Thống Gợi Ý Lai (Hybrid Recommendation System - HRS)
*   **Mục tiêu:** Khuyến nghị các mặt hàng người dùng có khả năng muốn mua/thuê nhất bằng cách kết hợp **hành vi tương tác** và **đặc trưng nhận dạng** của sản phẩm.
*   **Dữ liệu sử dụng (Đa thể thức):**
    1.  **Hành vi người dùng (User-Item Interaction):** Lịch sử mua sắm, thuê, hoặc click của từng người dùng.
    2.  **Ngôn ngữ (NLP):** Tên sản phẩm, mô tả, danh mục (áp dụng BERT hoặc TF-IDF để lượng giá hóa).
    3.  **Thị giác (Computer Vision):** Các đặc trưng ảnh (Vectors) được trích xuất từ mô-đun `rs_img_app`.
*   **Kiến trúc Thuật toán:** Sử dụng Mạng Nơ-ron Đồ Thị (Graph Neural Networks - GNN).
    *   **LightGCN:** Thuật toán truyền thông điệp cốt lõi để học mối quan hệ User-Item trên không gian đồ thị phức tạp.
    *   **CombiGCN, NGCF:** Các biến thể GCN mở rộng để đối chiếu hiệu suất.
    *   **BERT/NLP:** Trích xuất đặc trưng ngữ nghĩa chiều sâu từ văn bản.

## 3. Bảng So Sánh Chi Tiết

| Tiêu chí | `rs_img_app` (Truy xuất ảnh) | `hrs_system` (Gợi ý cá nhân hóa) |
| :--- | :--- | :--- |
| **Dữ liệu đầu vào** | Hình ảnh (Pixels trực quan) | Hành vi + Hình ảnh + NLP |
| **Kiến trúc cốt lõi** | CNN (MobileNet) + FAISS | Graph (LightGCN) + BERT |
| **Mệnh đề định hướng**| "Sản phẩm B nhìn giống hệt sản phẩm A" | "Người dùng X từng mua A nên dễ mua B" |
| **Mức độ phức tạp** | Thấp hơn, thiên về xử lý thị giác (CV) | Cao hơn, tối ưu hóa hệ thống điểm đa chiều |

## 4. Ý Nghĩa Của Các Hậu Tố (Suffixes) Dữ Liệu Đầu Vào
Các tệp thực thi trong `hrs_system` sử dụng các hậu tố để định danh luồng thông tin bổ trợ (Item Side Information) được cung cấp trong lúc chạy GCN:

*   **`_only_img`**: Chỉ áp dụng **Hình ảnh** làm thông tin bổ trợ cho các node sản phẩm trên đồ thị.
*   **`_tfidf_bert`**: Nạp **Văn bản (NLP)** làm dữ liệu bổ trợ (kết hợp TF-IDF thống kê phổ từ và phân tích ngữ nghĩa sâu BERT).
*   **`_bert_img`**: **Biến thể Hybrid (Mạng đa hợp - Mạnh nhất)**. Áp dụng đồng thời hai đặc trưng Văn bản (BERT) và Hình ảnh để tạo kết nối mạng toàn diện, mô hình có cái nhìn đầy đủ nhất về sản phẩm.
*   **Không hậu tố (`LightGCN.py`)**: Bản Collaborative Filtering cơ bản, chỉ sử dụng sự kiện tương tác hành động giữa User & Item, không khai thác yếu tố metadata.

## 5. Hướng Dẫn Thực Thi Đánh Giá (Evaluation)
*   Sử dụng Jupyter Notebook **`Eval_All.ipynb`** trong đường dẫn mã nguồn để chạy toàn diện các Metrics (Đo lường).
*   Notebook đã tích hợp tự động quy trình chấm điểm và so sánh các chỉ số `Recall`, `NDCG`.
*   **Khuyến nghị điểm xuất phát:** File **`LightGCN_bert_img.py`** – đây là biến thể thuật toán ổn định nhất trong họ GCN kết hợp tập dữ liệu đặc trưng hình ảnh và văn bản nhằm tối đa hoá hiệu suất học tập.