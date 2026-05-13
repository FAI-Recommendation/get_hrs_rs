# Cấu Trúc Tài Liệu - fashion-recommender-vto
Link github: https://github.com/AnhTaiNguyenKhac/Fashion-Product-Recommendation-System
Thư mục `docs/Note_fashion_prod/` lưu trữ các ghi chú phân tích chi tiết về kiến trúc mã nguồn và luồng di chuyển dữ liệu của thuật toán mạng nơ-ron đồ thị (GCN) áp dụng trong hệ thống gợi ý.

## Mục lục tài liệu

| File | Nội dung chi tiết |
| :--- | :--- |
| `01_Note.md` | Tổng quan về 2 module chính `rs_img_app` & `hrs_system`. |
| `02_SignAnalyze.md` | Giải nghĩa ý nghĩa tên các thư mục Dataset trong quá trình huấn luyện. |
| `03_QA-4.md` | Phân tích chi tiết sự khác biệt bên trong các thư mục Dataset. |
| `04_LightGCN.md` | Cấu trúc Source code của mô hình phân tích LightGCN. |
| `05_QA-1.md` | Sự khác biệt giữa code LightGCN xử lý đa phương thức (`only_img` vs `bert_img`). |
| `06_QA-2.md` | Cơ chế đọc dữ liệu hình ảnh dạng Vector của mô hình GCN. |
| `07_QA-3.md` | Cơ chế chuyển đổi (convert) pipeline từ file ảnh `.jpg` thành ma trận `.npz`. |
