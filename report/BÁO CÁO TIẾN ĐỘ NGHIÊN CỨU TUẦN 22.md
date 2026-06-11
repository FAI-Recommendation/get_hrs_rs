**BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU TUẦN 22**  
**Kính gửi:** Thầy Tiến sĩ Trần Trung Tín,

Em xin phép gửi thầy báo cáo tiến độ nghiên cứu tuần 22 của đề tài *"Integrating Multimodal Representations into Graph-based Fashion Recommender Systems"* (Capstone Project 3).

*   **Sinh viên thực hiện:** Hoàng Đình Quý Vũ (MSSV: 252805008)
*   **Thời gian báo cáo:** Tuần 22 (Từ 25/05/2026 đến 2/06/2026)

---
### 1. Các công việc đã thực hiện trong tuần:
Em đã hoàn thành viết bản thảo chi tiết và hoàn thiện cơ bản nội dung của cả **5 chương** trong báo cáo nghiên cứu:

*   **Chương 1 (Introduction):**
    *   Xây dựng hoàn chỉnh phần Lý do chọn đề tài (*Reason for choosing the topic*), chỉ rõ các thách thức đặc thù của lĩnh vực thời trang như tính mùa vụ và độ thưa thớt của dữ liệu giao dịch.
    *   Định nghĩa rõ ràng Mục tiêu tổng quát và các mục tiêu cụ thể, xác định giới hạn Phạm vi nghiên cứu (*Research Scope*) về mặt dữ liệu (VCR dataset), mô hình (GNNs) và các độ đo đánh giá.
*   **Chương 3 (Proposed Methods and Models):**
    *   Thiết kế và mô tả chi tiết luồng tiền xử lý dữ liệu (*Data Preprocessing Pipeline*) gồm: làm sạch dữ liệu, bộ lọc $N$-core ($N=5$) và thuật toán phân chia dữ liệu theo thời gian (*Per-user Temporal Split*) để tránh rò rỉ dữ liệu.
    *   Mô tả chi tiết cơ chế hoạt động, luồng lan truyền thông tin và tối ưu hóa của 3 mô hình chính: CombiGCN (adapted với item similarity graph), BM3 (Bootstrap contrastive learning) và FREEDOM (Decoupled graph structure learning).
*   **Chương 4 (Experiments and Results):**
    *   Trình bày chi tiết cấu hình thực nghiệm, siêu tham số huấn luyện (learning rate, L2 regularization, batch size, early stopping patience) để đảm bảo tính tái lập.
    *   Hệ thống hóa kết quả thực nghiệm thông qua Bảng so sánh hiệu năng tổng hợp và các biểu đồ radar, biểu đồ cột trực quan hóa hiệu quả khuyến nghị ở các độ sâu xếp hạng khác nhau ($K \in \{1, 5, 10, 20\}$).
    *   Thực hiện nghiên cứu loại bỏ (*Ablation study*) để phân tích vai trò độc lập của từng modality và tác động của cơ chế fusion.
*   **Chương 5 (Conclusion and Future Work):**
    *   Tổng hợp các kết luận thực nghiệm đắt giá (sự vượt trội của MobileNetV2 so với CLIP đối với việc học tương thích thời trang chi tiết cục bộ; tính ổn định của Late Fusion; sự khác biệt về hiệu năng của BM3 và CombiGCN ở các độ sâu xếp hạng khác nhau).
    *   Phân tích các hạn chế của đề tài về chi phí tính toán đồ thị và đề xuất hướng phát triển tương lai.
---

### 2. Kế hoạch tuần tiếp theo:
*   Tiến hành rà soát kỹ lưỡng lại toàn bộ các công thức toán học, kiểm tra lỗi chính tả và lỗi định dạng (formatting/linting errors) trong toàn văn văn bản để chuẩn bị cho bản nộp chính thức.
*   Tập trung hoàn thiện chương trình chạy thử nghiệm (Demo) trực quan và xây dựng Slide thuyết trình báo cáo đề tài.

---

### 3. Repository quản lý mã nguồn LaTeX:
Do tài khoản Overleaf bản miễn phí (free tier) đã hết giới hạn thời lượng biên dịch, em đã chuyển toàn bộ dự án LaTeX về máy cá nhân để biên dịch cục bộ và quản lý qua GitHub nhằm thuận tiện cho việc commit và theo dõi lịch sử chỉnh sửa.

Em xin gửi thầy liên kết kho lưu trữ (repository) mới của dự án để thầy tiện theo dõi tiến độ:  
👉 **Link GitHub:** [https://github.com/HoangVuSnape/01_Report_CD2_FashionRecommendation](https://github.com/HoangVuSnape/01_Report_CD2_FashionRecommendation)

Trân trọng,  
**Hoàng Đình Quý Vũ**