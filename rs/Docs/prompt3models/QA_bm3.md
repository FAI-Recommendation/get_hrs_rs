Dưới đây là tóm tắt toàn bộ cuộc trao đổi của chúng ta nãy giờ dưới dạng Hỏi - Đáp (Q&A) để bạn dễ dàng ôn lại trước khi báo cáo:

### 1. Kịch bản Thuyết trình Tổng thể

**Q: Làm sao để trình bày sơ đồ BM3 tinh chỉnh này cho thầy hướng dẫn dễ hiểu nhất?**
**A:** Mình đã cung cấp một kịch bản chi tiết đi từ dưới lên trên, có chia sẵn các nhịp ngắt và hành động [chỉ tay] vào từng khối:

* Bắt đầu từ ID Embeddings.
* Trình bày 2 nhánh song song: CF Branch (học tương tác qua đồ thị) và Modal Branch (xử lý đa phương thức không qua đồ thị).
* Chỉ vào bước Fusion để kết hợp đặc trưng.
* Nhấn mạnh vào khối tam giác Bootstrap Contrastive Learning.
* Chốt lại bằng hộp Loss tổng ở trên cùng.

### 2. Luồng Dữ liệu Học Tương phản

**Q: Đầu vào của 3 vòng tròn trong tam giác Bootstrap CL được lấy từ đâu?**
**A:** * **Online CF View:** Lấy từ nhánh đồ thị bên trái (đặc trưng tương tác).

* **Modal View:** Lấy từ lớp Linear Projectors bên phải (đặc trưng hình dáng/nội dung).
* **EMA Target View:** Lấy từ bản sao Item ID bị đóng băng ở dưới cùng bên trái (đóng vai trò mỏ neo).

### 3. Ý nghĩa Dòng chảy Gradient (Mũi tên lên Loss)

**Q: Tại sao lại có mũi tên từ $\hat{y}$ và từ 2 View (CF, Modal) chạy thẳng lên hộp Loss?**
**A:** Các mũi tên thể hiện luồng dữ liệu mang đi "tính điểm phạt":

* $\hat{y}$ mang lên để tính **BPR Loss** (giúp mô hình học cách xếp hạng gợi ý đúng).
* Đặc trưng từ CF View và Modal View mang lên để tính **Bootstrap CL Loss** (ép góc nhìn từ đồ thị và hình ảnh/văn bản phải đồng nhất với nhau).

### 4. Bản chất Đặc trưng Sản phẩm

**Q: Khác biệt giữa $h_i$ (item) và `item_target` là gì?**
**A:** * **$h_i$:** Là biểu diễn hoàn chỉnh cuối cùng (đã dung hợp đồ thị + đa phương thức). Nó liên tục thay đổi, dùng để tính điểm dự đoán $\hat{y}$.

* **`item_target`:** Chỉ là bản sao gốc bị đóng băng. Nó không dùng để dự đoán, mà làm "mỏ neo" đối chứng trong khối CL để mô hình không bị sụp đổ (collapse), chỉ cập nhật rất chậm qua EMA.

### 5. Chi tiết Trạm Dung hợp (Fusion)

**Q: Tại sao trên hình vẽ có tới 2 đường `item_cf` chĩa vào dấu (+) để ghép với `modal_emb`?**
**A:** Đó là lỗi hiển thị dư nét vẽ của sơ đồ gốc. Thực tế chỉ có MỘT `item_cf` đem **cộng** (không phải nối/concat) với `modal_emb` để tạo ra $h_i$. Đường `item_cf` còn lại chạy thẳng lên khối Online CF View.

### 6. Kích thước Dữ liệu Văn bản

**Q: Con số `[2194 x 768]` ở phần `text_feats` nghĩa là gì?**
**A:** * **2194:** Là tổng số lượng sản phẩm trong tập dữ liệu.

* **768:** Là số chiều vector mặc định khi dùng các mô hình ngôn ngữ lớn (như BERT) để đọc text.

### 7. Thiết kế Lớp Tuyến tính (Linear Projectors)

**Q: Tại sao không trích xuất văn bản ra thẳng 512 chiều cho khớp với ảnh, mà phải dùng thêm lớp Linear?**
**A:** Vì việc dùng lại các mô hình trích xuất khổng lồ đã train sẵn (như BERT ra 768, ResNet ra 512) là tiêu chuẩn. Việc cắm thêm một lớp Linear để ép 768 xuống 512 rẻ và tiết kiệm hơn hàng vạn lần so với việc train lại mô hình ngôn ngữ từ đầu. Hơn nữa, lớp Linear này còn đóng vai trò "phiên dịch" dữ liệu thô về chung một không gian thời trang.