# Phân Tích Source Code LightGCN

Tệp tin `LightGCN_only_img.py` (cũng như các cụm module `...bert_img.py` có liên kết tương đồng) là cấu trúc mã nguồn hạt nhân cốt lõi định hình và thiết lập Mạng Khuyến Nghị Dạng Đồ Thị (**Graph Neural Network Recommendation System**) trực thuộc ngôn ngữ Python sử dụng bộ thư viện TensorFlow 1.x.

## 1. Cấu Trúc Các Hàm (Methods) Chức Năng Hoạt Động Cốt Lõi

Khởi tạo mạng luới Class Đồ Thị GCN sẽ chứa thành phần tính toán Tensor chức năng:

*   **`__init__(self, ...)`**: Hàm khởi cấu trúc (Constructor). Trọng tâm đảm nhận nhiệm vụ khai báo danh sách bộ tham số tinh chỉnh, hình thành các trạm cấu trúc nạp tĩnh `placeholders` nhận mảng Users, Positive Items, Negative Items từ Dữ liệu. Xây dựng nền tảng hàm độ dốc Suy hao (Loss Function Thresholds) và tối ưu hóa biến mạng bằng thuật toán ngẫu nhiên chu kỳ (`Adam Optimizer`).
*   **`_init_weights(self)`**: Ma trận Vector cấp phát khởi điểm hệ thống đồ thị. Thuật toán này có thể xuất Random số nguyên Gaussian cho cấu trúc (Users, Items) hoặc phục chế kế thừa trạng thái trích xuất file (`Pretrained Embeddings`) tái khởi tạo chu kỳ.
*   **Các hàm `_split_A_hat`, `_split_SU_hat`, `_split_SI_hat`**: Đồ thị Mạng lưới Người Dùng – Trực thể là ma trận cực độ tĩnh mang đầy kích thước ma trận Không (`Sparse Tensor Zero Matrix`). Xử lý không khéo máy hệ thống sẽ xảy ra Memory Tràn RAM. Tập hợp Split này vận hành module cắt khối Ma Trận kích thước (Mini-batch) giúp quá trình học GPU trơn tru.
*   **`_create_combigcn_embed(self)`**: **Lõi Kernel (GNN Message Passing System)**. Mô đun lan truyền thông tin, học trọng số bằng cách đối chiếu Items với những Items bao quanh chúng trên cùng hệ Node. Điểm đột phá ở đặc trúc `only_img` chính là **bổ trợ tính tham số của ma trận sự tương đồng hình thể (`SI_fold_hat`)** cấu hình vào kết cấu thuật toán Vector nhằm tăng tỉ trọng liên kết.
*   **`create_bpr_loss(self, users, pos_items, neg_items)`**: Suy hao cá nhân phân bậc hạng Bayes (`BPR Loss`). Tối ưu mạng mô phỏng qua mục tiêu: Điểm trọng số từ *Sản phẩm Người Dùng từng Tương Tác* bắt buộc giữ Margin cực dốc đè lên tỉ số điểm *Sản phẩm Bị Loại Bỏ/Không Chú Ý*.
*   **Cấu trúc xử lý Luồng Data (`sample_thread`, `train_thread`...)**: Hỗ trợ xử lý bất đồng bộ cao tầng. 1 luồng CPU độc lập lấy số lượng dữ liệu `Random Sampling Data`, 1 lõi GPU chạy song song tính điểm bù cập nhật tham số. Đẩy ngưỡng FPS học thuật tối ưu đáng kể.
*   **Trung Tâm Thực Thi `if __name__ == '__main__':`**: Module điều hướng chu trình Pipeline bao gồm Nạp -> Khởi tạo -> Lặp `epoch` -> So sánh dốc sai phân dự đoán `TensorBoard` -> Kiểm Định Metrics (`Recall`, `NDCG`, `MRR`...) ở từng thập vòng huấn luyện.

## 2. Ý Nghĩa Của Các Hệ Số Command Line Thực Thi (Hyper-parameters)

Mã truyền tải các khối Parameter lúc lệnh run:

```bash
python LightGCN_only_img.py --dataset vcr1p_img_m1_img --Ks [1,5,10,20] --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 8192 --epoch 1000
```

*   **`--dataset vcr1p_img_m1_img`**: Quy định phân luồng đọc thư mục đầu vào. Dữ liệu trích xuất mẫu 1% `VCR` tiền xử lý `M1 MobileNet`.
*   **`--Ks [1,5,10,20]`**: Chạy định mức K (Số lượng List phân loại phản hồi). Đơn vị Top `1, 5, 10, 20` sản phẩm mang xếp hạng cao trúng vị trí người dùng yêu cầu nhất phục vụ Audit Metrics.
*   **`--regs [1e-5]`**: Mức giới hạn tinh giản bộ máy phạt L2 Penalty Threshold (`0.00001`). Hệ thống không được "Học Thuộc Data – Overfitting" và neo kích thước Weight Embeddings theo không gian cực trị của AI.
*   **`--embed_size 64`**: Hệ thống mảng số cấp không gian 64 Chiều ma trận tính trạng cấp cho mỗi tài khoản và mặt hàng.
*   **`--layer_size [64,64,64]`**: Độ dày kiến trúc hệ mạng Đồ Thị. List gán với 3 phân cấp **Layer GCN Convolution** sẽ lan truyền điểm tín hiệu: (Layer 1 - Lan truyền hàng mua / Layer 2 - Lan truyền bạn bè mua / Layer 3 - Lan truyền sở thích chung từ cộng đồng mua). Mỗi chuỗi rút 64 Vector. 
*   **`--lr 0.001`**: Mật độ độ sải học thuật – Tốc độ nhảy Gradient điều chỉnh sau đợt test fail (Learning Rate).
*   **`--batch_size 8192`**: Tức thì lấy cấu trúc cập nhật khung hình `8192 Model Tương tác/Batch`. Do GCN phải duyệt qua kết cấu ma trận Sparse, tính GPU sẽ hiệu năng ổn định hơn nếu dung nạp ma trận đủ to làm đầy số lõi phần cứng của vRAM xử lý.
*   **`--epoch 1000`**: Tố tụng chu trình duyệt chéo toàn hệ thống Data trong 1000 lần lặp. Hệ thống cấu trúc nhúng phương hướng **Early Stopping**: Giảm kẹt xe chu kỳ Model tự Check-point cắt ngang vòng đời khi điểm chỉ số Accuracy bị đóng băng ở cực đại.