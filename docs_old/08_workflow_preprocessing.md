# Workflow Tiền Xử Lý `preprocess-rs-vcr.ipynb`

Notebook này không phải notebook train model. Nhiệm vụ của nó là biến dữ liệu VCR gốc thành các file trung gian chuẩn hoá để các mô hình khuyến nghị phía sau như LightGCN, NGCF, CombiGCN đọc tiếp.

## Quick view 

Mình đang đọc notebook để dựng lại luồng xử lý end-to-end, rồi sẽ trả lại cho bạn một workflow ngắn gọn kèm sơ đồ để thấy rõ nó tạo ra gì từ dữ liệu gốc đến các file đầu ra.



Notebook này là một pipeline tiền xử lý dữ liệu cho bài toán recommendation, không phải notebook train model. Nó lấy dữ liệu VCR thô, làm sạch, đồng bộ các bảng, rồi xuất ra các file trung gian để các notebook mô hình như LightGCN, NGCF, CombiGCN dùng tiếp. File bạn đang hỏi nằm ở preprocess-rs-vcr.ipynb.

Luồng end-to-end của nó là:

1. Đọc 3 nguồn dữ liệu gốc: outfits.csv, picture_triplets.csv, user_activity_triplets.csv.
2. Làm sạch outfits: parse các cột dạng list, bỏ dòng thiếu name, thay missing description bằng chuỗi mặc định.
3. Làm sạch pictures và transactions: loại các outfit.id không hợp lệ, rồi giữ lại các giao dịch chỉ còn khớp với dữ liệu ảnh.
4. Tạo bảng tương tác chuẩn hóa gồm user_id_original, item_id_original, time và lưu thành dataset_VCR.csv.
5. Lấy mẫu dữ liệu và lọc n-core trong process_data, sau đó map lại ID user/item về dạng số liên tiếp, lưu thêm dataset_VCR_0.5_42_10.csv, user_list.txt, item_list.txt, intersection_user.txt.
6. Sắp xếp theo user và thời gian, tạo rank theo lịch sử tương tác, rồi chia train/test theo từng user và lưu train.txt, test.txt.
7. Ghép metadata item từ outfits với nhóm ảnh theo outfit.id, tạo image_list.txt.
8. Tạo items_features.csv với 3 loại đặc trưng:
   - feature1: name + outfit_tags
   - feature2: description
   - feature3: embedding ảnh, rồi gộp theo mean, weighted hoặc max và giảm chiều bằng PCA.
9. Có thêm phần debug để đọc lại items_features.csv, parse vector embedding và kiểm tra BERT/text embeddings.

Nếu bạn muốn, mình có thể vẽ lại luồng này thành một sơ đồ khối gọn hơn, kiểu “raw data → cleaning → interaction files → feature files → model input”.

## 1. Mục tiêu cuối cùng

Sau khi chạy xong notebook, hệ thống tạo ra 3 nhóm đầu ra chính:

1. File tương tác người dùng - item: `dataset_VCR.csv`, `train.txt`, `test.txt`, `user_list.txt`, `item_list.txt`, `intersection_user.txt`.
2. File mô tả item: `items_features.csv`.
3. File phụ trợ cho ảnh: `image_list.txt`.

Nói ngắn gọn, notebook này làm 3 việc liên tiếp: làm sạch dữ liệu, chuẩn hoá ID, rồi tạo feature cho item.

## 2. Dữ liệu đầu vào

Notebook đọc 3 file chính:

1. `outfits.csv`: metadata của outfit, gồm tên, mô tả, tag, nhóm, và các trường mô tả item.
2. `picture_triplets.csv`: thông tin ảnh của từng outfit, gồm `picture.id`, `file_name`, `displayOrder`, `outfit.id`.
3. `user_activity_triplets.csv`: log tương tác người dùng với outfit theo thời gian thuê.

## 3. Luồng xử lý tổng quát

```mermaid
flowchart TD
   A[Raw VCR files\noutfits.csv\npicture_triplets.csv\nuser_activity_triplets.csv] --> B[Load data with pandas]
   B --> C[Parse list-like columns\noutfit_tags, tag_categories]
   C --> D[Clean outfits\ndrop rows with missing name\nfill missing description]
   D --> E[Filter picture rows\nremove invalid outfit.id]
   D --> F[Filter transaction rows\nremove invalid outfit.id]
   E --> G[Keep only valid outfit.id in pictures]
   F --> G
   G --> H[Build normalized interaction table\nuser_id_original, item_id_original, time]
   H --> I[Save dataset_VCR.csv]
   I --> J[Sample + n-core filter\nprocess_data(random_percent, n_core)]
   J --> K[Map original IDs to consecutive IDs]
   K --> L[Save dataset_VCR_{sample}_{seed}_{n_core}.csv]
   K --> M[Create user_list.txt\nitem_list.txt\nintersection_user.txt]
   K --> N[Split train/test by user rank]
   N --> O[Save train.txt\ntest.txt]
   D --> P[Merge outfits with item mapping]
   E --> Q[Group pictures by outfit.id]
   P --> R[Build item feature table]
   Q --> R
   R --> S[Create feature1\nname + tags]
   R --> T[Create feature2\ndescription]
   R --> U[Create feature3 from image embeddings\nmean/weighted/max + PCA]
   S --> V[Save items_features.csv]
   T --> V
   U --> V
```

## 4. Ý nghĩa từng bước

### 4.1 Làm sạch `outfits`

Notebook đầu tiên đọc và kiểm tra định dạng file `outfits.csv`, sau đó:

1. Parse các cột dạng chuỗi biểu diễn list như `outfit_tags` và `tag_categories`.
2. Loại các dòng thiếu `name` vì đây là thông tin lõi để mô tả item.
3. Điền `description` bị thiếu bằng giá trị mặc định `No description available` để tránh mất dữ liệu đầu vào cho mô hình văn bản.

Kết quả của đoạn này là một bảng outfits sạch hơn, dùng làm nền để ghép với các bảng khác.

### 4.2 Làm sạch `picture_triplets`

Sau khi có danh sách `outfit.id` bị loại do thiếu `name`, notebook lọc `picture_triplets.csv` để bỏ các ảnh thuộc outfit không còn hợp lệ.

Mục đích của bước này là giữ cho ảnh và metadata item luôn khớp nhau, tránh trường hợp ảnh còn tồn tại nhưng item gốc đã bị loại.

### 4.3 Làm sạch `user_activity_triplets`

Với bảng tương tác người dùng, notebook:

1. Đọc dữ liệu theo dấu phân tách `;`.
2. Chuyển `rentalPeriod.start` và `rentalPeriod.end` sang kiểu datetime.
3. Loại các giao dịch có `outfit.id` không còn tồn tại trong tập hợp hợp lệ.

Sau đó tạo bảng chuẩn hoá gồm:

1. `user_id_original`
2. `item_id_original`
3. `time`

Bảng này được lưu thành `dataset_VCR.csv`.

### 4.4 Chuẩn hoá và lấy mẫu dữ liệu

Hàm `process_data()` thực hiện phần preprocessing cho tập tương tác:

1. Lấy mẫu ngẫu nhiên một phần dữ liệu theo `random_percent`.
2. Lọc n-core theo số tương tác tối thiểu của user.
3. Map lại ID user và item từ mã gốc sang ID số liên tiếp bắt đầu từ 0.
4. Lưu file kết quả ra `dataset_VCR_{random_percent}_{random_state}_{n_core}.csv`.

Đây là bước rất quan trọng vì các mô hình GNN thường cần ID dạng số liên tục để tạo embedding và sparse matrix.

### 4.5 Tạo danh sách user, item và interaction

Từ dữ liệu đã map ID, notebook tạo các file sau:

1. `user_list.txt`: ánh xạ giữa `user_id_original` và `user_id`.
2. `item_list.txt`: ánh xạ giữa `item_id_original` và `item_id`.
3. `intersection_user.txt`: với mỗi user, liệt kê toàn bộ item mà user đã tương tác.

Những file này thường được các pipeline đồ thị hoặc batch loader dùng để dựng ma trận tương tác.

### 4.6 Chia train/test theo từng user

Notebook không chia ngẫu nhiên toàn bộ dataset, mà chia theo từng user dựa trên thứ tự thời gian:

1. Thêm cột `rank` bằng `groupby("user_id").cumcount() + 1`.
2. Tính ngưỡng train cho mỗi user theo tỷ lệ 80%.
3. Các tương tác đầu được đưa vào `train_interactions`.
4. Các tương tác còn lại được đưa vào `test_interactions`.

Sau đó lưu ra `train.txt` và `test.txt`.

Cách chia này giữ cho mỗi user đều xuất hiện trong cả train lẫn test, phù hợp với bài toán recommendation theo lịch sử hành vi.

### 4.7 Xây dựng feature cho item

Notebook ghép `outfits` với `item_df` để tạo metadata item, rồi ghép thêm dữ liệu ảnh từ `picture_triplets`.

Sau khi merge, nó tạo `items_features.csv` với 3 feature chính:

1. `feature1`: kết hợp `name` và `outfit_tags`.
2. `feature2`: lấy trực tiếp từ `description`.
3. `feature3`: vector embedding ảnh.

Riêng `feature3` được tạo bằng cách:

1. Xây danh sách đường dẫn tới file embedding `.npy` của từng ảnh.
2. Load embedding của từng ảnh trong cùng outfit.
3. Gộp embedding theo một trong ba cách: mean, weighted, hoặc max.
4. Giảm chiều bằng PCA để đưa vector về kích thước phù hợp hơn.

Đây là phần giúp notebook chuyển từ metadata thô sang feature đa phương thức cho mô hình gợi ý.

## 5. Output cuối cùng và vai trò của chúng

1. `dataset_VCR.csv`: dữ liệu tương tác gốc đã được chuẩn hoá.
2. `dataset_VCR_0.5_42_10.csv` hoặc các bản tương tự: dữ liệu đã sample, lọc n-core và map ID.
3. `user_list.txt`, `item_list.txt`, `intersection_user.txt`: file phụ trợ cho mapping và adjacency/interactions.
4. `train.txt`, `test.txt`: tập dữ liệu dùng cho huấn luyện và đánh giá.
5. `image_list.txt`: danh sách ảnh theo item.
6. `items_features.csv`: file feature cuối cùng cho item, gồm text và image embedding.

## 6. Kết luận ngắn

Nếu nhìn theo pipeline, notebook này đi từ dữ liệu thô VCR sang dữ liệu đã sẵn sàng cho mô hình khuyến nghị đa phương thức. Nó không tạo ra model, mà chuẩn bị toàn bộ đầu vào mà model cần.

Nếu bạn muốn, mình có thể viết tiếp một bản “sơ đồ 1 trang” rất ngắn để bạn dán vào tài liệu, hoặc chuyển phần này thành flowchart ASCII dễ nhìn hơn cho README.