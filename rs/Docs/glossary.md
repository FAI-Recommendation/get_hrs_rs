# Glossary – Thuật Ngữ Hệ Thống Gợi Ý Thời Trang

> Tổng hợp toàn bộ thuật ngữ kỹ thuật xuất hiện trong tài liệu dự án.  
> Sắp xếp theo nhóm chủ đề để dễ tra cứu.

---

## 1. Mô hình & Thuật toán (Models & Algorithms)

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **LightGCN** | Graph Convolutional Network phiên bản nhẹ — chỉ dùng neighbor aggregation, bỏ qua feature transformation và nonlinear activation. Là backbone chính của `hrs_system`. |
| **NGCF** | Neural Graph Collaborative Filtering — tiền thân của LightGCN, dùng thêm nonlinear layer khi propagate embedding. |
| **CombiGCN** | Biến thể GCN kết hợp đa ma trận kề (interaction + similarity) để học embedding. |
| **GCN / GNN** | Graph Convolutional Network / Graph Neural Network — mạng nơ-ron hoạt động trên dữ liệu dạng đồ thị. Mỗi node học embedding từ các node láng giềng xung quanh. |
| **BPR Loss** | Bayesian Personalized Ranking Loss — hàm loss chuẩn cho bài toán recommendation: ép score của (user, positive item) > score của (user, negative item). |
| **MobileNetV2** | CNN nhẹ của Google, dùng để trích xuất feature ảnh. Output: vector 1280 chiều (`include_top=False, pooling='avg'`). Được train trên ImageNet. |
| **CLIP** | Contrastive Language-Image Pretraining — model của OpenAI, train trên 400 triệu cặp (ảnh, caption). Output: vector 512 chiều. Tốt hơn MobileNetV2 cho fashion vì hiểu ngữ nghĩa ảnh. |
| **BERT** | Bidirectional Encoder Representations from Transformers — model NLP của Google. Dùng `bert-base-uncased` để tạo text embedding 768 chiều từ `feature1 + feature2`. |
| **YOLOv5s** | You Only Look Once v5 (small) — model object detection, dùng để crop vùng quần áo ra khỏi ảnh gốc trước khi tạo embedding (module `rs_img_app`). |
| **FAISS** | Facebook AI Similarity Search — thư viện tìm kiếm vector tương đồng nhanh. Dùng trong `rs_img_app` cho tính năng "tìm sản phẩm tương tự". |
| **Adam Optimizer** | Thuật toán tối ưu gradient descent thích nghi, dùng để cập nhật weight trong LightGCN mỗi epoch. |
| **Early Stopping** | Dừng training sớm khi metric (Recall, NDCG) không cải thiện sau N epoch liên tiếp, tránh overfitting. |
| **ViT (Vision Transformer)** | Kiến trúc Transformer áp dụng cho ảnh — chia ảnh thành các patch rồi xử lý như chuỗi token. CLIP dùng ViT-Base làm encoder ảnh. |

---

## 2. Kiến trúc Hệ thống (System Architecture)

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **rs_img_app** | Module truy xuất ảnh tương tự (Visual Similarity Search) — dùng YOLOv5s + MobileNetV2 + FAISS. Không cần lịch sử user. |
| **hrs_system** | Hybrid Recommendation System — module gợi ý cá nhân hóa dùng GNN kết hợp hành vi user + ảnh + text. |
| **Pipeline** | Chuỗi các bước xử lý dữ liệu theo thứ tự từ raw data → model input, mỗi bước output là input của bước tiếp theo. |
| **Preprocessing** | Giai đoạn tiền xử lý: làm sạch data, mapping ID, tạo feature — **không phải** training model. |
| **Offline Preprocessing** | Xử lý trước khi training (tính embedding, tạo similarity matrix) để lúc train không phải tính lại từ đầu. |
| **Cold-start** | Vấn đề khi user hoặc item quá mới, không có đủ lịch sử tương tác để model đưa ra gợi ý chính xác. |

---

## 3. Dữ liệu & Dataset (Data)

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **VCR** | Vibrent Clothes Rental Dataset — dataset thuê quần áo từ Kaggle, gồm 3 file CSV: `outfits.csv`, `picture_triplets.csv`, `user_activity_triplets.csv`. |
| **RTR** | Rent The Runway — dataset thời trang khác, dùng để đối chứng kết quả. |
| **Outfit** | Một bộ trang phục trong hệ thống, tương ứng với một "item" trong RS. Mỗi outfit có thể có 4-5 ảnh. |
| **user_activity_triplets** | File ghi log giao dịch thuê đồ: `customer.id`, `outfit.id`, `rentalPeriod.start`, `rentalPeriod.end`. |
| **picture_triplets** | File thông tin ảnh: `picture.id`, `outfit.id`, `file_name`, `displayOrder`. |
| **displayOrder** | Thứ tự hiển thị ảnh trong một outfit. `displayOrder == 0` là ảnh toàn thân chính diện — ảnh đại diện tốt nhất cho outfit. |
| **Canonical View** | Ảnh `displayOrder == 0` — góc nhìn chuẩn, chứa nhiều Global Features nhất (màu sắc, form dáng, phong cách tổng thể). |
| **n-core filtering** | Lọc bỏ user có số tương tác < n. Ví dụ `n_core=5` loại bỏ user tương tác < 5 lần. Kỹ thuật chuẩn trong RS để tránh cold-start và sparse matrix. |
| **Random Sampling** | Lấy ngẫu nhiên `k%` dữ liệu (`random_percent=1.0` = dùng 100%). Dùng để giảm kích thước dataset khi thử nghiệm. |
| **User-centric Sampling** | Lấy mẫu theo đơn vị user (giữ toàn bộ giao dịch của user được chọn), thay vì random từng dòng. Tránh trường hợp user chỉ còn 1 giao dịch trong tập test. |
| **UNIX Timestamp** | Số giây tính từ 01/01/1970 UTC. Cột `time` trong `dataset_VCR.csv` được convert sang dạng này để dễ sort và so sánh. |
| **Temporal Split** | Chia train/test theo thứ tự thời gian (không random): 80% giao dịch đầu của mỗi user → train, 20% cuối → test. |
| **Per-user Temporal Split** | Mỗi user được chia train/test độc lập — đảm bảo mọi user đều xuất hiện trong cả 2 tập. |
| **ID Mapping** | Ánh xạ ID gốc (string dài như `outfit.abc123`) sang ID số nguyên liên tiếp (0, 1, 2, ...) vì GNN cần integer index để tạo embedding matrix. |

---

## 4. Embedding & Feature (Đặc trưng)

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **Embedding** | Vector số học biểu diễn một đối tượng (user, item, ảnh, text) trong không gian chiều cao. Khoảng cách giữa 2 vector phản ánh mức độ tương đồng. |
| **Image Embedding** | Vector đặc trưng của ảnh, sinh bởi CNN (MobileNetV2 → 1280-D hoặc CLIP → 512-D). Lưu trong file `.npy`. |
| **Text Embedding** | Vector đặc trưng của văn bản, sinh bởi BERT (768-D). Mã hóa ngữ nghĩa của `feature1 + feature2`. |
| **feature1** | Đặc trưng text ngắn của item: `name + outfit_tags`. Dùng cho TF-IDF similarity. |
| **feature2** | Đặc trưng text dài của item: `description`. Dùng cho BERT similarity. |
| **feature3** | Đặc trưng ảnh của item: vector embedding ảnh đã được gộp (mean/weighted/max) và giảm chiều PCA → lưu trong `items_features.csv`. |
| **items_features.csv** | File đặc trưng item, gồm 4 cột: `item_id`, `feature1`, `feature2`, `feature3`. Là input trực tiếp của `load_data.py`. |
| **`.npy` file** | File NumPy binary lưu một array — dùng để lưu embedding vector của từng ảnh. Tên file = `<picture_id>.npy`. |
| **Mean Fusion (M1)** | Gộp embedding nhiều ảnh bằng cách lấy trung bình: `avg(emb1, emb2, ...)`. Baseline, ổn định nhất. |
| **Weighted Fusion (M2)** | Gộp embedding có trọng số, ảnh `displayOrder=0` được ưu tiên hơn. Hợp lý khi outfit có nhiều ảnh. |
| **Max Fusion (M3)** | Gộp embedding bằng cách lấy giá trị lớn nhất theo từng chiều: `max(emb1, emb2, ...)`. Capture đặc trưng nổi bật nhất. |
| **PCA** | Principal Component Analysis — giảm chiều vector từ 1280 → 768 để đồng nhất với BERT embedding và giảm noise. |
| **L2 Normalization** | Chuẩn hóa vector về độ dài 1 (unit norm), thực hiện trước PCA và trước khi tính cosine similarity. |
| **Pretrained Embedding** | Embedding được load từ file checkpoint đã train sẵn, thay vì khởi tạo random. |
| **preprocess_input()** | Hàm chuẩn hóa ảnh đúng với cách model được train: MobileNetV2 dùng `(pixel / 127.5) - 1` → range [-1, 1], **không** dùng `/ 255.0`. |

---

## 5. Đồ thị & Ma trận (Graph & Matrix)

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **Adjacency Matrix (Ma trận kề)** | Ma trận biểu diễn mối liên kết giữa các node trong đồ thị. Phần tử `A[i][j] = 1` nghĩa là node i và node j có cạnh nối. |
| **Sparse Matrix** | Ma trận thưa — hầu hết phần tử bằng 0. Dùng định dạng CSR (`scipy.sparse.csr_matrix`) để tiết kiệm bộ nhớ. |
| **`.npz` file** | File NumPy nén lưu sparse matrix — cache các ma trận kề để lần sau load nhanh, không tính lại. |
| **Interaction Matrix (R)** | Ma trận User-Item interaction: `R[u][i] = 1` nếu user u đã tương tác với item i. Xây từ `train.txt`. |
| **SI Matrix** | Item-Item Similarity Matrix — ma trận tương đồng giữa các item dựa trên ảnh, text, hoặc multimodal. Threshold thường 0.5. |
| **SU Matrix** | User-User Similarity Matrix — ma trận tương đồng giữa các user dựa trên Jaccard similarity của lịch sử mua sắm. |
| **Social Adjacency Matrix** | Ma trận quan hệ xã hội giữa users (nếu có `social_trust.txt`). |
| **Normalized Symmetric Matrix** | Ma trận kề đã được chuẩn hóa đối xứng: `D^(-1/2) * A * D^(-1/2)`. Cần thiết để propagation trong GCN ổn định. |
| **Cosine Similarity** | Đo góc giữa 2 vector: `cos(θ) = (A·B) / (|A|·|B|)`. Giá trị 1 = giống nhau hoàn toàn, 0 = vuông góc. Threshold 0.5 trong pipeline. |
| **Message Passing** | Cơ chế GCN: mỗi node tổng hợp thông tin từ các node láng giềng qua từng layer. Sau L layer, node biết về "bạn bè của bạn bè". |
| **s_img_similarity_adj_mat.npz** | Cache ma trận tương đồng ảnh giữa các item — sinh từ `feature3` lần đầu chạy `load_data.py`. |
| **s_bert_item_similarity_adj_mat.npz** | Cache ma trận tương đồng text BERT giữa các item — sinh từ `feature2`. |
| **s_multimodal_similarity_adj_mat.npz** | Cache ma trận tương đồng multimodal (ảnh + text) — dùng cho chế độ `bert_img`. |
| **s_tfidf_item_similarity_adj_mat.npz** | Cache ma trận tương đồng TF-IDF — sinh từ `feature1`. |

---

## 6. Chiến lược Fusion (Kết hợp đa thể thức)

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **Early Fusion** | Kết hợp embedding ảnh và text **trước** khi tính similarity (concatenate hoặc average). Kết quả kém vì "pha loãng" đặc trưng riêng của từng modality. |
| **Late Fusion** | Tính similarity riêng cho ảnh và text, sau đó **gộp kết quả** ở bước cuối. Tốt hơn Early Fusion vì giữ nguyên tín hiệu từng modality. |
| **Weight Attention Fusion** | Học trọng số tự động để kết hợp similarity ảnh và text — model tự quyết định khi nào tin vào ảnh hơn hay text hơn. Tốt nhất trong 3 chiến lược. |
| **Multimodal** | Đa thể thức — kết hợp nhiều loại dữ liệu: hình ảnh (CV) + văn bản (NLP) + hành vi user. |
| **only_img** | Chế độ LightGCN chỉ dùng Image Similarity Matrix làm đồ thị bổ trợ. Đọc `similar_items_adj` từ vị trí 8 trong `get_norm_adj_mat()`. |
| **bert_img** | Chế độ LightGCN dùng Multimodal Similarity Matrix (ảnh + BERT text). Đọc `similar_items_adj` từ vị trí 7. |
| **tfidf_bert** | Chế độ dùng text làm đồ thị bổ trợ, kết hợp TF-IDF thống kê từ và BERT ngữ nghĩa sâu. |

---

## 7. Metric Đánh giá (Evaluation Metrics)

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **Recall@K** | Tỷ lệ item liên quan được gợi ý trong top-K kết quả. `Recall@10 = 0.4` nghĩa là 40% item user thực sự thích xuất hiện trong top 10 gợi ý. |
| **NDCG@K** | Normalized Discounted Cumulative Gain — đánh giá thứ tự xếp hạng: item liên quan xuất hiện càng cao trong top-K càng tốt. |
| **Top-K Recommendation** | Gợi ý K sản phẩm phù hợp nhất cho user. Pipeline đánh giá với K = 1, 3, 5, 10. |
| **Hit Ratio** | Tỷ lệ user có ít nhất 1 item đúng trong top-K gợi ý. |
| **MRR** | Mean Reciprocal Rank — trung bình `1/rank` của item đúng đầu tiên trong danh sách gợi ý. |

---

## 8. Kỹ thuật Lập trình (Code & Engineering)

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **Batch Processing** | Xử lý nhiều ảnh cùng lúc (32 ảnh/batch) thay vì từng ảnh, tận dụng GPU parallelism — nhanh hơn ~10x. |
| **GPU Memory Growth** | Cài đặt TensorFlow chỉ cấp VRAM theo nhu cầu thực tế, tránh chiếm hết 6GB VRAM ngay khi khởi động. |
| **Lazy Caching** | Pattern try/except trong `load_data.py`: thử load file `.npz` từ disk, nếu không có thì tính rồi lưu. Từ lần 2 trở đi load nhanh. |
| **uv** | Package manager Python nhanh (thay thế pip) — dùng để cài thư viện trong project này. |
| **CUDA** | Compute Unified Device Architecture — nền tảng tính toán song song của NVIDIA. PyTorch cần bản `+cu124` để chạy trên GPU RTX 4050. |
| **`.venv`** | Virtual environment — môi trường Python độc lập của project, chứa tất cả thư viện cài vào. |
| **`ast.literal_eval()`** | Parse chuỗi Python như `"['tag1', 'tag2']"` thành list thực sự. Dùng để đọc cột `outfit_tags`, `tag_categories`. |
| **`on_bad_lines='skip'`** | Tham số `pd.read_csv()` — bỏ qua dòng lỗi định dạng thay vì raise exception. |
| **`sep=';'`** | File CSV của VCR dùng dấu chấm phẩy làm delimiter, không phải dấu phẩy thông thường. |
| **`cumcount()`** | Hàm pandas đếm thứ tự trong mỗi nhóm (groupby) — dùng để tạo cột `rank` theo thứ tự thời gian của mỗi user. |
| **enumerate(start=0)** | Tạo ID số liên tiếp bắt đầu từ 0 khi mapping — GNN cần 0-indexed integer ID. |
| **`torch.no_grad()`** | Context manager tắt tính toán gradient khi inference — tiết kiệm bộ nhớ và tăng tốc. |

---

## 9. File & Thư mục quan trọng

| Tên file/thư mục | Vai trò |
|-----------------|---------|
| `slipt_10k_sample.py` | Script lấy mẫu 10k giao dịch từ VCR đầy đủ, lọc displayOrder==0. |
| `copy_images_10k.py` | Copy 2223 ảnh chính (displayOrder==0) vào thư mục `images_2223_main/`. |
| `get_embedding_MBNV2_optimized.py` | Tạo 2223 file `.npy` embedding ảnh bằng MobileNetV2, batch=32, GPU. |
| `get_embedding_CLIP.py` | Tạo embedding ảnh bằng CLIP ViT-B/32, output 512-D, không cần PCA. |
| `preprocess-rs-vcr-10k.ipynb` | Notebook tiền xử lý: clean data → ID mapping → tạo feature → `items_features.csv`. |
| `preprocess_vcr_10k.py` | Script Python thuần chuyển đổi từ notebook trên — chạy được không cần Jupyter. |
| `load_data.py` | Đọc `items_features.csv` + `train.txt` → tính toán → lưu tất cả file `.npz`. Chạy tự động lần đầu khi train model. |
| `LightGCN_bert_img.py` | File model mạnh nhất — kết hợp ảnh + BERT text trong đồ thị GCN. |
| `Eval_All.ipynb` | Notebook đánh giá tổng hợp tất cả metric (Recall, NDCG) cho tất cả model. |
| `embeddings_10k/` | Thư mục chứa 2223 file `.npy` — embedding ảnh MobileNetV2. |
| `embeddings_clip_10k/` | Thư mục chứa 2223 file `.npy` — embedding ảnh CLIP. |
| `images_2223_main/` | 2223 ảnh `.jpg` đã chọn lọc (displayOrder==0). |
| `output_10k_sample/` | Thư mục gốc chứa tất cả output của pipeline 10k. |

---

## 10. Ký hiệu & Hậu tố (Naming Convention)

| Ký hiệu | Ý nghĩa |
|---------|---------|
| `vcr` | Dataset Vibrent Clothes Rental |
| `rtr` | Dataset Rent The Runway |
| `_10k` | File CSV là bản sample ~10k giao dịch (không đổi schema gốc) |
| `1p`, `5p` | Lấy mẫu 1% hoặc 5% dữ liệu gốc |
| `img_m1` | Image embedding dùng Mean Fusion |
| `img_m2` | Image embedding dùng Weighted Fusion |
| `img_m3` | Image embedding dùng Max Fusion |
| `_detect` | Ảnh đã qua YOLOv5s crop trước khi tạo embedding |
| `_only_img` | Model chỉ dùng Image Similarity làm side information |
| `_bert_img` | Model dùng cả BERT text + Image (multimodal) |
| `_tfidf_bert` | Model dùng TF-IDF + BERT text |
| `_late_fusion` | Chiến lược gộp similarity ở bước cuối |
| `_weightattention` | Chiến lược dùng attention học trọng số tự động |
| `_multiheadattention` | Chiến lược dùng multi-head attention |
| `_pca` | Có bước giảm chiều PCA trước khi tính similarity |
| `s_*.npz` | File cache sparse matrix (prefix `s_` = sparse) |
