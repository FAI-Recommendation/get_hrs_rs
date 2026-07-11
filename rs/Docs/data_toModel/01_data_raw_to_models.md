# Tổng Hợp Quy Trình Xử Lý Dữ Liệu: Từ Raw Data Đến Đầu Vào 3 Model

Tài liệu này hệ thống hóa toàn bộ luồng dữ liệu (Data Pipeline), bắt đầu từ tập dữ liệu thô (VCR dataset) ban đầu, qua các bước chọn lọc mẫu **10k giao dịch**, tiền xử lý làm sạch, xây dựng embeddings đặc trưng, và cuối cùng là chuẩn bị các file input để cấp trực tiếp cho 3 mô hình học máy: **CombiGCN**, **BM3**, và **FREEDOM**.

---

## 1. Các File Tài Liệu Tham Chiếu Gốc (Source Docs)
Các bước xử lý và phân tích này được tổng hợp từ các file tài liệu hiện có trong project:
*   **Tổng quan và phân tích luồng:** 
    *   [01_pipeline_tong_quan.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/01_pipeline_tong_quan.md)
    *   [00_tong_quan_pipeline.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/data_raw_toEnd/00_tong_quan_pipeline.md)
*   **Chi tiết thuật toán trích chọn & split:**
    *   [02_thuat_toan_split_10k.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/02_thuat_toan_split_10k.md)
    *   [01_chi_tiet_tung_buoc.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/data_raw_toEnd/01_chi_tiet_tung_buoc.md)
    *   [01_data_raw_pipeline.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/models_pyg_training_latest/01_data_raw_pipeline.md)
*   **Vai trò của Embeddings & Cấu trúc Model:**
    *   [05_embeddings_10k_vai_tro.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/05_embeddings_10k_vai_tro.md)
    *   [02_model_architectures.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/models_pyg_training_latest/02_model_architectures.md)
    *   [03_data_embeddings.md](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/Docs/models_pyg_training_latest/03_data_embeddings.md)

---

## 2. Sơ Đồ Tổng Quan Quy Trình (Pipeline Flowchart)

```
RAW DATA (VCR Dataset thô)
 ├── user_activity_triplets.csv (~64k giao dịch, ~1600+ users)
 ├── picture_triplets.csv       (~50k ảnh, nhiều ảnh/outfit)
 └── outfits.csv                (Thông tin metadata tên, tag của outfit)
         │
         ▼
 BƯỚC 1: Lấy mẫu 10k (`slipt_10k_sample.py`)
 ├── Lọc ảnh chính diện (displayOrder == 0)
 ├── Loại bỏ outfit null name hoặc thiếu ảnh chính diện
 ├── Chọn user có >= 3 giao dịch, shuffle ngẫu nhiên và cộng dồn đạt ~10k giao dịch
 │
 ├──▶ Output 1: user_activity_triplets_10k.csv  (~10,000 giao dịch)
 ├──▶ Output 2: picture_triplets_10k.csv        (2,223 ảnh chính của outfit)
 └──▶ Output 3: outfits_10k.csv                 (2,223 outfits tương ứng)
         │
         ▼
 BƯỚC 2: Tiền xử lý sâu, N-Core, ID Mapping (`preprocess_vcr_10k.py`)
 ├── Loại bỏ triệt để các outfit bị lỗi null name hoặc thiếu ảnh trong folder embeddings_10k/
 ├── Áp dụng N-Core Filter (N_CORE = 5): Chỉ giữ lại các user có tối thiểu 5 tương tác
 ├── Thực hiện ID Mapping: Chuyển chuỗi ID (string) thành ID số nguyên liên tục bắt đầu từ 0
 │
 ├──▶ Output 1: dataset_VCR_1.0_42_5.csv        (9,455 giao dịch, 553 users, 2,194 items)
 ├──▶ Output 2: user_list.txt / item_list.txt   (File ánh xạ ID gốc và ID mới)
 └──▶ Output 3: intersection_user.txt           (Tất cả tương tác đã được ánh xạ ID mới)
         │
         ▼
 BƯỚC 3: Phân chia tập Train/Test theo thời gian per-user (80/20 Split)
 ├── Đánh chỉ số (rank) theo thời gian giao dịch của từng user (cũ nhất -> mới nhất)
 ├── 80% giao dịch cũ hơn của user đó đưa vào tập Train
 ├── 20% giao dịch mới hơn đưa vào tập Test
 │
 ├──▶ train.txt (7,350 interactions - chứa đủ 553 users)
 └──▶ test.txt  (2,105 interactions - chứa đủ 553 users)
```

---

## 3. Chi Tiết Từng Bước Biến Đổi Dữ Liệu

### BƯỚC 1: Trích lọc mẫu 10k giao dịch (`slipt_10k_sample.py`)
Mục tiêu là trích xuất một mẫu đại diện **~10k giao dịch** từ tập dữ liệu thô ban đầu (~64k giao dịch) nhưng vẫn giữ được cấu trúc dữ liệu nguyên bản.

1.  **Lọc ảnh chính diện (`pick_main_picture_rows`):**
    *   Mỗi outfit trong `picture_triplets.csv` có 4-5 ảnh chụp ở các góc khác nhau (mặt trước, chụp lưng, chất liệu vải, mác áo).
    *   Script chỉ chọn ảnh có cột `displayOrder == 0` (ảnh model mặc toàn thân góc trực diện). Đây là góc chứa nhiều thông tin tổng thể nhất (Global Features), giúp quá trình tạo embedding của mô hình học sâu (như MobileNetV2 hay CLIP) được tối ưu và hội tụ nhanh hơn.
    *   Loại bỏ trùng lặp bằng cách giữ dòng ảnh đầu tiên của mỗi `outfit.id`.
2.  **Liên kết dữ liệu (Inner Join):**
    *   Thực hiện Inner Join giữa 3 bảng: Giao dịch, Ảnh đã lọc (`displayOrder == 0`), và Outfit Metadata.
    *   Chỉ giữ các giao dịch thỏa mãn: outfit có ảnh chính diện hợp lệ và có tên (`name` không bị rỗng/null).
3.  **Lấy mẫu hướng người dùng (User-centric Sampling):**
    *   **Vì sao không lấy ngẫu nhiên theo dòng?** Nếu lấy ngẫu nhiên, một số user chỉ còn lại 1-2 giao dịch lẻ tẻ. Khi chia Train/Test, các giao dịch này có thể rơi hoàn toàn vào Test (không có lịch sử ở Train để model học preference), hoặc rơi hoàn toàn vào Train (không có gì để đánh giá ở Test).
    *   **Thuật toán:**
        1. Lọc ra danh sách các user hợp lệ (có tối thiểu 3 giao dịch trong tập dữ liệu gốc).
        2. Trộn ngẫu nhiên (shuffle) danh sách user này với seed cố định (`random_state=42`).
        3. Duyệt qua từng user, lấy toàn bộ lịch sử giao dịch của họ cộng dồn lại cho đến khi tổng số dòng đạt tối thiểu 10,000 giao dịch thì dừng.
4.  **Kết xuất đầu ra Bước 1:**
    *   `user_activity_triplets_10k.csv`: ~10.000 tương tác.
    *   `picture_triplets_10k.csv` & `outfits_10k.csv`: Có đúng **2,223** hàng (đảm bảo mỗi outfit có đúng 1 ảnh đại diện).

---

### BƯỚC 2: Làm sạch sâu, lọc N-Core và ID Mapping (`preprocess_vcr_10k.py`)
Mục tiêu của bước này là chuẩn hóa dữ liệu sang dạng ma trận thưa và đảm bảo các user đều có đủ tín hiệu (signals) để mô hình học.

1.  **Lọc N-Core (`N_CORE = 5`):**
    *   Chỉ giữ lại các user có tối thiểu 5 giao dịch trong mẫu 10k vừa trích chọn. Các user có ít hơn 5 giao dịch sẽ bị loại bỏ hoàn toàn.
    *   *Hiệu ứng dây chuyền (Indirect Item Removal):* Khi loại bỏ các user có quá ít tương tác, một số outfit (item) vốn chỉ được tương tác bởi các user bị loại này cũng sẽ không còn tương tác nào nữa. Do đó, số lượng item giảm nhẹ từ **2,223** xuống còn **2,194** (mất 29 items).
2.  **ID Mapping (Ánh xạ ID sang số nguyên):**
    *   Các ID gốc của user và item trong database là các chuỗi ký tự (string index, ví dụ: `"outfit-abc123xyz"`).
    *   Các thư viện đồ thị (PyTorch Geometric, LightGCN) yêu cầu ID đầu vào phải là số nguyên liên tục bắt đầu từ 0 (`0, 1, 2, ...`).
    *   Tạo dictionary ánh xạ và lưu file lưu trữ:
        *   `user_list.txt`: ánh xạ `user_id_original` ➔ `user_id` (từ 0 đến 552)
        *   `item_list.txt`: ánh xạ `item_id_original` ➔ `item_id` (từ 0 đến 2193)
3.  **Kết xuất đầu ra Bước 2:**
    *   `dataset_VCR_1.0_42_5.csv`: Tập dữ liệu sạch cuối cùng gồm 9,455 giao dịch, 553 users, và 2.194 items.

---

### BƯỚC 3: Phân chia tập Train/Test theo thời gian per-user (80/20)
Mục tiêu của bước này là chia dữ liệu tương tác để phục vụ cho việc huấn luyện và kiểm thử mô hình.

*   **Tại sao không chia ngẫu nhiên (Random Split)?**
    Nếu chia ngẫu nhiên trên toàn bộ tập dữ liệu, một số user có thể bị đẩy hết tương tác sang tập Test (không học được preference ở Train) hoặc sang tập Train (không đánh giá được hiệu năng ở Test). Đồng thời, chia ngẫu nhiên sẽ gây rò rỉ dữ liệu (Data Leakage) vì mô hình có thể dùng hành vi tương lai để dự đoán quá khứ.
*   **Giải pháp - Per-user Temporal Split:**
    1. Với mỗi user riêng biệt, sắp xếp tất cả giao dịch của họ theo thứ tự thời gian tăng dần.
    2. Đánh số thứ tự (`rank`) từ 1 đến $N$ (với $N$ là tổng số giao dịch của user đó).
    3. Tính toán ngưỡng phân chia cho từng user: $\text{threshold} = \lfloor N \times 0.8 \rfloor$.
    4. Các giao dịch có $\text{rank} \le \text{threshold}$ được đưa vào `train.txt` (80% tương tác đầu tiên của mỗi user).
    5. Các giao dịch có $\text{rank} > \text{threshold}$ được đưa vào `test.txt` (20% tương tác cuối cùng của mỗi user).
*   **Đặc điểm quan trọng:** Cả `train.txt` và `test.txt` đều chứa chính xác **553 users**.
*   **Tỉ lệ thực tế:** Vì phép làm tròn xuống (`.astype(int)`), tỷ lệ phân chia tổng thể thực tế là khoảng **77.7% Train / 22.3% Test** (do các user có số giao dịch lẻ bị làm tròn xuống tập Train).

---

## 4. Xử Lý Và Tạo Embeddings Đặc Trưng (Modality Processing)

Để cấp dữ liệu đa phương tiện cho mô hình đa phương thức (Multimodal Recommendation), các thông tin văn bản (Text) và hình ảnh (Image) của các outfit được xử lý như sau:

### Hình ảnh (Image Features)
*   **Nguồn gốc:** Thư mục `embeddings_10k/` chứa 2,223 file dạng `.npy` (mỗi file ứng với phần ID sau dấu "." của `picture.id`). 
*   **Trích xuất đặc trưng hình ảnh:**
    *   Sử dụng mạng pre-trained **MobileNetV2** để trích xuất đặc trưng từ ảnh gốc, tạo ra vector có kích thước ban đầu là `(1280,)`.
    *   Qua notebook tiền xử lý, các vector được gom lại (1 vector/outfit) và áp dụng phương pháp giảm chiều dữ liệu **PCA** để chuyển từ `1280` chiều xuống còn **`768` chiều** nhằm giảm tải tính toán.
    *   Giá trị vector này được lưu dưới dạng chuỗi string trong cột `feature3` của file `items_features.csv`.
    *   *Lưu ý:* Nếu sử dụng mô hình **CLIP**, vector ảnh trích xuất sẽ trực tiếp có độ dài **`512` chiều** và không cần thực hiện giảm chiều bằng PCA.

### Văn bản (Text Features)
*   **Nguồn gốc:** Cột tên và mô tả outfit trong `outfits.csv`.
*   **Trích xuất đặc trưng văn bản:**
    *   `feature1`: Kết hợp `name` + `outfit_tags`. Thường được vector hóa bằng phương pháp **TF-IDF** hoặc mô hình **BERT**.
    *   `feature2`: Cột `description` của outfit (đầu vào cho BERT).

### Lưu trữ đặc trưng trong file `items_features.csv`:
```csv
item_id,feature1,feature2,feature3
0,"Summer Dress casual beach","Light fabric...", "[0.123, -0.456, ...]"
```

---

## 5. Cấu Trúc Thư Mục Data Hoàn Chỉnh Cấp Cho Các Model

Sau khi tiền xử lý hoàn tất, tất cả các file dữ liệu được gom vào một thư mục dữ liệu đích (ví dụ `clip_10k_sample/` hoặc `mbnv2_10k_sample/`). Thư mục này là nguồn cấp dữ liệu duy nhất cho cả 3 mô hình huấn luyện:

```
clip_10k_sample/
├── train.txt                 # File chứa tương tác huấn luyện (553 users)
├── test.txt                  # File chứa tương tác kiểm thử (553 users)
├── items_features.csv        # Chứa text & vector đặc trưng ảnh của 2,194 items
├── image_embeddings.npy      # Ma trận lưu sẵn vector ảnh của items, shape (2194, img_dim)
├── text_embeddings.npy       # Ma trận lưu sẵn vector chữ của items, shape (2194, txt_dim)
└── s_*.npz                   # (Tự động sinh) Bộ nhớ cache các ma trận kề đã chuẩn hóa
```

### Bộ nhớ cache các ma trận kề và tương tự (`s_*.npz`)
Khi chạy code huấn luyện lần đầu, class `Data` trong file `load_data.py` sẽ tự động tính toán các ma trận kề chuẩn hóa đối xứng ($D^{-0.5} A D^{-0.5}$) và lưu lại (cache) thành các file `.npz` để tăng tốc độ load cho các lần chạy sau:

1.  `s_interaction_adj_mat.npz`: Ma trận kề đồ thị lưỡng phân User-Item.
2.  `s_tfidf_item_similarity_adj_mat.npz`: Ma trận tương đồng cosine giữa các item dựa trên đặc trưng văn bản TF-IDF (với ngưỡng cosine similarity > 0.5).
3.  `s_img_similarity_adj_mat.npz`: Ma trận tương đồng cosine giữa các item dựa trên đặc trưng ảnh MobileNetV2/CLIP (ngưỡng > 0.5).
4.  `s_multimodal_similarity_adj_mat.npz`: Kết hợp đa phương thức (ví dụ: trung bình cộng trọng số $0.5 \times \text{text\_sim} + 0.5 \times \text{img\_sim}$).

---

## 6. Cách Dữ Liệu Được Đưa Vào Từng Model

Cả 3 model đều sử dụng chung cơ chế **BPR Sampling** (Bayesian Personalized Ranking) trong quá trình huấn luyện: với mỗi Batch, lấy ngẫu nhiên danh sách các bộ ba `(User, Pos_Item, Neg_Item)` từ `train.txt` để tối ưu hóa việc xếp hạng.

Tuy nhiên, mỗi mô hình sẽ tiếp nhận và xử lý ma trận đặc trưng theo các cách khác nhau dựa trên tham số cấu hình `sim_type` (`none` / `img_only` / `tfidf` / `multimodal` / `multimodal_attention`):

### 1. Mô hình CombiGCN (Dual-Graph GCN)
*   **Luồng hoạt động:** 
    Kết hợp luồng lan truyền thông tin trên 2 đồ thị song song: **Đồ thị tương tác User-Item** và **Đồ thị tương đồng Item-Item** (`similarity_adj`).
*   **Cơ chế nhận đầu vào:**
    *   Đọc ma trận kề tương tác `interaction_adj` từ `s_interaction_adj_mat.npz`.
    *   Đọc ma trận tương đồng `similarity_adj` từ file tương ứng với cấu hình `sim_type` (ví dụ: `s_img_similarity_adj_mat.npz` nếu `sim_type=img_only`).
    *   *Trường hợp đặc biệt:* Nếu `sim_type=none`, mô hình sẽ tắt nhánh Item-Item, hoạt động như một mô hình **LightGCN** thuần túy.

### 2. Mô hình BM3 (Bootstrap Latent Representations)
*   **Luồng hoạt động:** 
    Không xây dựng đồ thị tương đồng Item-Item dạng tĩnh. Thay vào đó, nó đưa trực tiếp vector đặc trưng thô (`image_feats`, `text_feats`) vào các mạng chiếu (Projectors) để ánh xạ về cùng không gian biểu diễn (modal embedding), sau đó cộng trực tiếp vào Collaborative Filtering (CF) embedding thu được từ LightGCN. Mô hình sử dụng Contrastive Loss (Bootstrap CL) để tự giám sát giữa các view đa phương thức.
*   **Cơ chế nhận đầu vào:**
    *   Đọc `train.txt`, `test.txt` để xây dựng đồ thị tương tác.
    *   Đọc trực tiếp ma trận `image_embeddings.npy` và `text_embeddings.npy` làm đặc trưng đầu vào cho các Projector.

### 3. Mô hình FREEDOM (Frozen and Denoising Graphs)
*   **Luồng hoạt động:**
    Sử dụng các đặc trưng đa phương tiện (`image_feats`, `text_feats`) để tính toán ma trận tương đồng, từ đó xây dựng một đồ thị K-lân cận gần nhất (K-Nearest Neighbors - kNN graph) cho Item ngay khi khởi tạo mô hình và đóng băng đồ thị này (`FROZEN`). Trong lúc huấn luyện, embeddings sẽ được lan truyền trên đồ thị kNN tĩnh này để tạo ra biểu diễn nội dung (`item_emb_content`), sau đó kết hợp với biểu diễn tương tác thông qua hàm loss tương phản InfoNCE.
*   **Cơ chế nhận đầu vào:**
    *   Đọc `image_embeddings.npy` và `text_embeddings.npy` để chạy thuật toán xây dựng đồ thị kNN ban đầu.
    *   Nhận đồ thị tương tác `interaction_adj` và thực hiện khử nhiễu (denoising) song song.
