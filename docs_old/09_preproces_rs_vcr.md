Tôi sẽ tạo một todo list ngắn để theo dõi các bước: xác nhận chỗ embedding được dùng, giải thích vai trò, và xác nhận doc là preprocessing hay training.

Created 3 todos

Progress: đã ghi TODOs và xác nhận các bước. Tiếp theo là phần giải thích chi tiết.

- Embedding "V2" (ví dụ thư mục embeddings_MBNV2_full) là vector đặc trưng ảnh đã tính trước (MobileNetV2). Notebook tải các file .npy tương ứng với mỗi `picture.id` và dùng chúng để tạo `feature3` cho mỗi outfit/item.
- Cách dùng trong `preprocess-rs-vcr.ipynb`: load các embedding ảnh cho từng ảnh của một outfit → gộp thành embedding của outfit bằng một trong ba cách (mean, weighted theo displayOrder, hoặc max) → (tùy chọn) chuẩn hoá/giảm chiều bằng PCA → lưu vào cột `feature3` trong `items_features.csv`.
- Ý nghĩa trong việc làm data: embedding V2 cung cấp biểu diễn số (dense vector) của thông tin hình ảnh — tức là chuyển thông tin ảnh thành feature mà mô hình khuyến nghị có thể tiêu thụ. Nó:
  - cho phép kết hợp đa phương thức (text + image) trong mô hình (CombiGCN, LightGCN_bert_img...),
  - dùng để tính ma trận tương đồng ảnh hoặc bổ sung thông tin item khi dữ liệu tương tác nghèo,
  - giúp so khớp item theo nội dung hình ảnh khi train/evaluate.
- Mối liên hệ với 08_workflow_preprocessing.md: file này là tài liệu mô tả bước tiền xử lý (preprocessing). Nó không phải là script train; nhưng là tài liệu quan trọng mô tả cách tạo ra dữ liệu đầu vào cho training — gồm cả embedding V2 được gộp vào `items_features.csv`. Nói ngắn: doc là hướng dẫn/ghi chép quá trình chuẩn bị dữ liệu mà training sẽ dùng.
- Nếu bạn cần kiểm chứng đoạn code cụ thể, mình có thể trích đoạn nơi notebook:
  - tạo `embedding_paths`, 
  - hàm `load_mean_embeddings`/`load_weighted_embeddings`/`load_max_embeddings`, và 
  - chỗ gọi PCA → tất cả nằm trong preprocess-rs-vcr.ipynb.
- Muốn mình làm gì tiếp theo? (chọn 1)
  1. Trích chính xác các cell code liên quan và dán vào doc.  
  2. Thêm một đoạn "how-to" ngắn trong 08_workflow_preprocessing.md mô tả cách tái tạo hoặc thay embedding (ví dụ đường dẫn, kích thước vector, PCA dim).  
  3. Vẽ sơ đồ dòng dữ liệu rõ hơn chỉ cho phần embedding → item features.