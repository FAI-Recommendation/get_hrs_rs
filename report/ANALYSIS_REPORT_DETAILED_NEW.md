# BÁO CÁO PHÂN TÍCH THỰC NGHIỆM CHI TIẾT
## Đánh giá hệ thống gợi ý thời trang đa phương thức (24 Model Configs)

*   **Đề tài Capstone Project 3:** Integrating Multimodal Representations into Graph-based Fashion Recommender Systems
*   **Dữ liệu thực nghiệm:** Trích xuất tự động và tính toán từ log của 24 runs Weights & Biases (WandB) lưu tại `data_evaluate/data_wandb`.
*   **Dataset:** 553 users, 2.194 items, 7.350 tương tác huấn luyện, 2.105 tương tác kiểm thử (~3.8 test items/user).

---

## 1. So Sánh Hiệu Năng Giữa Các Mô Hình Tối Ưu (Best-vs-Best)

Báo cáo phân tích hiệu năng của cấu hình tối ưu nhất (lựa chọn theo chỉ số **NDCG@10**) cho 3 kiến trúc mô hình chính: **BM3**, **CombiGCN**, và **FREEDOM**. 

Do đặc thù tập dữ liệu thời trang kiểm thử có mật độ tương tác rất thưa (trung bình chỉ ~3.8 lượt tương tác thực tế/user), chúng tôi chọn **K=5** làm chỉ số đánh giá thực tế chính (độ sâu recommend bằng 1.3 lần kích thước ground truth thực tế, giúp phân biệt rõ ràng chất lượng xếp hạng của mô hình mà không bị bão hòa sớm) và giữ **K=10** song song để đối chiếu với các nghiên cứu tiền nhiệm.

### Bảng 1: So sánh hiệu năng các cấu hình tối ưu của từng mô hình

| Thứ hạng | Mô hình | Cấu hình tối ưu | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR@5 | MRR@10 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **BM3** | `mbnv2 · multimodal` | **0.0162** | **0.0186** | **0.0166** | **0.0250** | **0.0105** | **0.0078** | **0.0259** | **0.0291** |
| **2** | **CombiGCN** | `mbnv2 · multimodal` | 0.0137 | 0.0175 | 0.0127 | 0.0244 | 0.0090 | **0.0081** | 0.0242 | 0.0290 |
| **3** | **FREEDOM** | `mbnv2 · multimodal_attention` | 0.0084 | 0.0088 | 0.0105 | 0.0132 | 0.0054 | 0.0031 | 0.0118 | 0.0124 |

#### Mã nguồn LaTeX bảng biểu so sánh Best-vs-Best:
```latex
\begin{table}[htbp]
  \centering
  \caption{So sánh hiệu năng giữa các cấu hình tối ưu của từng mô hình}
  \label{tab:best_vs_best}
  \begin{tabular}{llcccccccc}
    \toprule
    \textbf{Rank} & \textbf{Model} & \textbf{Config} & \textbf{NDCG@5} & \textbf{NDCG@10} & \textbf{Recall@5} & \textbf{Recall@10} & \textbf{Prec@5} & \textbf{Prec@10} & \textbf{MRR@5} \\
    \midrule
    1 & BM3 & mbnv2(multimodal) & \textbf{0.0162} & \textbf{0.0186} & \textbf{0.0166} & \textbf{0.0250} & \textbf{0.0105} & 0.0078 & \textbf{0.0259} \\
    2 & CombiGCN & mbnv2(multimodal) & 0.0137 & 0.0175 & 0.0127 & 0.0244 & 0.0090 & \textbf{0.0081} & 0.0242 \\
    3 & FREEDOM & mbnv2(mm\_attn) & 0.0084 & 0.0088 & 0.0105 & 0.0132 & 0.0054 & 0.0031 & 0.0118 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Nhận xét & Điểm dị biệt chính:
*   **BM3 chiếm ưu thế xếp hạng sâu:** Cấu hình `bm3 · mbnv2 · multimodal` dẫn đầu tuyệt đối ở các chỉ số NDCG@5 (đạt **0.0162**, cao hơn CombiGCN **18.25%** và gấp **1.93 lần** FREEDOM), NDCG@10, Recall@5, và Recall@10. Khả năng học tự giám sát tương phản (contrastive learning) đa góc nhìn giúp BM3 định hình không gian nhúng có tính phân tách và xếp hạng sâu cực kỳ tốt.
*   **Điểm dị biệt tại K=1:** Khi phân tích sâu kết quả tại K=1 (từ file summary gốc), **CombiGCN** (`mbnv2 · multimodal`) đạt NDCG@1 = **0.0127** và Precision@1 = **0.0127**, hoàn toàn tương đương với **BM3** (NDCG@1 = **0.0127**, Precision@1 = **0.0127**). Hơn thế nữa, tại K=10, CombiGCN lại nhỉnh hơn BM3 về Precision@10 (**0.0081** so với **0.0078**). Điều này cho thấy với các nhiệm vụ chỉ gợi ý đúng 1 vật phẩm duy nhất hoặc ưu tiên độ chính xác tuyệt đối trong Top-10, CombiGCN dựa trên cấu trúc đồ thị tích hợp trực tiếp vẫn là một đối thủ cạnh tranh rất mạnh.
*   **Hiệu năng FREEDOM tụt sâu:** FREEDOM đạt hiệu năng rất thấp trên tất cả các metrics. NDCG@10 của FREEDOM (đạt **0.0088**) thấp hơn BM3 đến **52.69%**. Cơ chế học cấu trúc đồ thị phân tách của FREEDOM dường như không khớp nối tốt với các đặc trưng đa phương thức trên tập dữ liệu quy mô nhỏ.

---

## 2. Tác Động Của Visual Encoder (RQ1: CLIP vs MobileNetV2)

Để trả lời câu hỏi nghiên cứu về việc bộ trích xuất đặc trưng hình ảnh nào phù hợp hơn cho miền thời trang, chúng tôi lập bảng so sánh hiệu năng NDCG@10 khi thay đổi Visual Encoder giữa **CLIP** và **MobileNetV2 (MBNv2)** đối với 12 cặp cấu hình tương ứng.

### Bảng 2: So sánh hiệu năng NDCG@10 giữa CLIP và MobileNetV2

| Mô hình | Phương pháp Similarity (Sim Type) | NDCG@10 (CLIP) | NDCG@10 (MBNv2) | Mức cải thiện của MBNv2 (%) |
| :--- | :--- | :---: | :---: | :---: |
| **BM3** | `img_only` | 0.0129 | 0.0150 | +16.29% |
| **BM3** | `tfidf` | **0.0158** | 0.0149 | -6.06% (CLIP thắng) |
| **BM3** | `multimodal` | 0.0142 | **0.0186** | **+30.84%** |
| **BM3** | `multimodal_attention` | 0.0059 | 0.0101 | +72.24% |
| **CombiGCN** | `img_only` | 0.0155 | 0.0085 | -45.20% (CLIP thắng) |
| **CombiGCN** | `tfidf` | 0.0077 | 0.0071 | -7.52% (CLIP thắng) |
| **CombiGCN** | `multimodal` | 0.0174 | **0.0175** | +0.67% |
| **CombiGCN** | `multimodal_attention` | 0.0153 | 0.0151 | -1.59% (CLIP thắng) |
| **FREEDOM** | `img_only` | 0.0060 | 0.0062 | +3.67% |
| **FREEDOM** | `tfidf` | 0.0049 | 0.0031 | -36.06% (CLIP thắng) |
| **FREEDOM** | `multimodal` | 0.0049 | 0.0081 | +63.71% |
| **FREEDOM** | `multimodal_attention` | 0.0033 | **0.0088** | **+165.29%** |

#### Mã nguồn LaTeX bảng biểu so sánh Visual Encoders:
```latex
\begin{table}[htbp]
  \centering
  \caption{Tác động của Visual Encoder lên hiệu năng NDCG@10}
  \label{tab:encoder_comparison}
  \begin{tabular}{llccc}
    \toprule
    \textbf{Model} & \textbf{Sim Type} & \textbf{NDCG@10 (CLIP)} & \textbf{NDCG@10 (MBNv2)} & \textbf{Improvement (\%)} \\
    \midrule
    BM3 & img\_only & 0.0129 & 0.0150 & +16.29\% \\
    BM3 & tfidf & \textbf{0.0158} & 0.0149 & -6.06\% \\
    BM3 & multimodal & 0.0142 & \textbf{0.0186} & \textbf{+30.84\%} \\
    BM3 & mm\_attn & 0.0059 & 0.0101 & +72.24\% \\
    \midrule
    CombiGCN & img\_only & 0.0155 & 0.0085 & -45.20\% \\
    CombiGCN & tfidf & 0.0077 & 0.0071 & -7.52\% \\
    CombiGCN & multimodal & 0.0174 & \textbf{0.0175} & +0.67\% \\
    CombiGCN & mm\_attn & 0.0153 & 0.0151 & -1.59\% \\
    \midrule
    FREEDOM & img\_only & 0.0060 & 0.0062 & +3.67\% \\
    FREEDOM & tfidf & 0.0049 & 0.0031 & -36.06\% \\
    FREEDOM & multimodal & 0.0049 & 0.0081 & +63.71\% \\
    FREEDOM & mm\_attn & 0.0033 & \textbf{0.0088} & \textbf{+165.29\%} \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Nhận xét & Lý giải chuyên môn:
1.  **MobileNetV2 tạo ra các Best-config:** Ở các cấu hình tối ưu của cả 3 mô hình (BM3 multimodal, CombiGCN multimodal, FREEDOM attention), MobileNetV2 đều mang lại hiệu năng cao nhất (BM3: **0.0186** vs 0.0142, FREEDOM: **0.0088** vs 0.0033). CLIP hoàn toàn thất thế ở các cấu hình đỉnh này.
2.  **Lý giải sự vượt trội của MobileNetV2 (Miền thời trang đặc thù):** 
    *   **Fine-grained representation (Học biểu diễn mịn):** MobileNetV2 được pre-train trên ImageNet với mục tiêu phân loại vật thể cụ thể, do đó mạng cực kỳ nhạy cảm với các đặc trưng visual cục bộ tầm thấp và trung như họa tiết (*textures*), đường viền quần áo (*shapes*), hoa văn, túi áo, cúc bấm... Đây lại là những yếu tố quyết định tính tương thích và tương đồng trong phối đồ thời trang.
    *   **Hạn chế của CLIP semantic abstraction (Trừu tượng hóa ngữ nghĩa):** CLIP được huấn luyện contrastive quy mô lớn trên các cặp ảnh-văn bản từ Internet để học ngữ nghĩa cảnh quan khái quát (*high-level concept semantics*). Do đó, CLIP biểu diễn ảnh dưới dạng một khái niệm ngữ nghĩa tổng thể (ví dụ: "một cô gái mặc váy đi trên phố"). Khi đưa vào GNN gợi ý sản phẩm thời trang thưa thớt, embedding của CLIP bị thiếu hụt các đặc trưng chi tiết cục bộ cần thiết, dẫn đến việc gợi ý kém sắc nét hơn.
3.  **Dị biệt thú vị của CLIP ở các cấu hình phụ:**
    *   CLIP chiến thắng rõ rệt ở cấu hình `combigcn · img_only` (NDCG@10 đạt **0.0155** so với **0.0085** của MobileNetV2).
    *   CLIP tfidf của BM3 đạt NDCG@10 cực kỳ cao (**0.0158**), vượt cả bản CLIP multimodal. Điều này chỉ ra rằng khi không có Late Fusion tối ưu, cơ chế biểu diễn ngữ nghĩa của CLIP có khả năng bổ trợ rất tốt cho nhãn văn bản TF-IDF, tạo ra sự căn chỉnh không gian biểu diễn tự nhiên hơn so với MobileNetV2.

---

## 3. Đánh Giá Các Phương Pháp Kết Hợp Đặc Trưng (RQ2: Ablation Study)

Để làm rõ vai trò của việc kết hợp các nguồn dữ liệu đa phương thức, chúng tôi thực hiện nghiên cứu loại bỏ (ablation study) trên 4 cơ chế liên kết đặc trưng bằng cách cố định visual encoder tốt nhất là **MobileNetV2**.

### Bảng 3: Ablation Study hiệu năng NDCG@10 của các sim_type (MBNv2)

| Mô hình | Chỉ sử dụng Visual (`img_only`) | Chỉ sử dụng Text (`tfidf`) | Late Fusion (`multimodal`) | Late Fusion + Attention (`multimodal_attention`) |
| :--- | :---: | :---: | :---: | :---: |
| **BM3** | 0.0150 | 0.0149 | **0.0186** | 0.0101 (Tụt **45.74%**) |
| **CombiGCN** | 0.0085 | 0.0071 | **0.0175** | 0.0151 (Tụt **13.73%**) |
| **FREEDOM** | 0.0062 | 0.0031 | 0.0081 | **0.0088** (Tăng **8.26%**) |

#### Mã nguồn LaTeX bảng biểu Ablation Study:
```latex
\begin{table}[htbp]
  \centering
  \caption{Ablation Study trên các cơ chế kết hợp đặc trưng (Visual Encoder: MobileNetV2)}
  \label{tab:ablation_study}
  \begin{tabular}{lcccc}
    \toprule
    \textbf{Model} & \textbf{img\_only} & \textbf{tfidf} & \textbf{multimodal} & \textbf{multimodal\_attention} \\
    \midrule
    BM3 & 0.0150 & 0.0149 & \textbf{0.0186} & 0.0101 \\
    CombiGCN & 0.0085 & 0.0071 & \textbf{0.0175} & 0.0151 \\
    FREEDOM & 0.0062 & 0.0031 & 0.0081 & \textbf{0.0088} \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Nhận xét & Lý giải chuyên môn:
1.  **Late Fusion (`multimodal`) mang lại hiệu năng tối ưu:** Đối với cả BM3 và CombiGCN, việc kết hợp đồng thời văn bản (text) và hình ảnh (visual) mang lại kết quả xếp hạng tốt nhất. Chỉ số NDCG@10 của `multimodal` vượt trội rõ rệt so với các cấu hình đơn phương thức (`img_only`, `tfidf`), chứng minh thông tin mô tả sản phẩm bằng văn bản và đặc trưng visual bổ trợ rất tốt cho nhau.
2.  **Nghịch lý Attention (`multimodal_attention`):**
    *   Thêm cơ chế attention học trọng số phi tuyến tính làm hiệu năng của **BM3 tụt dốc nghiêm trọng** đến **45.74%** (từ 0.0186 xuống 0.0101) và làm **CombiGCN giảm 13.73%**.
    *   Chỉ duy nhất **FREEDOM** cải thiện nhẹ hiệu năng (tăng **8.26%** từ 0.0081 lên 0.0088) khi tích hợp attention.
    *   **Lý giải:** BM3 và CombiGCN vốn dĩ đã có các cơ chế lan truyền đồ thị và tự giám sát tối ưu hóa ma trận kề đa phương thức rất ổn định. Việc cố gắng học thêm một lớp trọng số attention phi tuyến tính trên tập dữ liệu giao dịch quy mô nhỏ (~9.4k tương tác) đã tạo ra hiện tượng **quá khớp (overfitting)** nghiêm trọng và bổ sung nhiều nhiễu trong quá trình tối ưu hóa. FREEDOM là mô hình yếu, việc học cấu trúc đồ thị bị phân tách có thể được lớp attention hỗ trợ điều hướng thông tin tốt hơn, giúp nó cải thiện nhẹ, nhưng vẫn không đủ để so sánh với hai mô hình còn lại.

---

## 4. Phân Tích Hành Vi Hội Tụ Và Quá Khớp (Overfitting)

Chúng tôi trích xuất giá trị epoch đạt hiệu năng tối ưu trên tập validation (`best_epoch`), so sánh với tổng số epoch thiết lập (1000) và đối chiếu giá trị Loss huấn luyện tại `best_epoch` so với khi kết thúc huấn luyện để đánh giá hành vi hội tụ và quá khớp.

### Bảng 4: Thống kê quá trình hội tụ và biến động loss của các best-config

| Dòng mô hình | Cấu hình tối ưu | Epoch đạt tối ưu (`best_epoch`) | Train Loss tại `best_epoch` | Train Loss tại Epoch 1000 | Tỷ lệ giảm Loss sau Best Epoch (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BM3** | `mbnv2 · multimodal` | **720** | 0.03688 | 0.02904 | -21.26% |
| **CombiGCN** | `mbnv2 · multimodal` | **280** | 0.04055 | 0.01186 | **-70.75%** |
| **FREEDOM** | `mbnv2 · multimodal_attention` | **960** | 0.46834 | 0.46978 | +0.31% (Dao động nhẹ) |

#### Mã nguồn LaTeX bảng biểu phân tích quá trình hội tụ:
```latex
\begin{table}[htbp]
  \centering
  \caption{Thống kê quá trình hội tụ và biến động Loss của các cấu hình tốt nhất}
  \label{tab:convergence_analysis}
  \begin{tabular}{lccccc}
    \toprule
    \textbf{Model} & \textbf{Best Config} & \textbf{Best Epoch} & \textbf{Loss@Best} & \textbf{Loss@1000} & \textbf{Loss Decrease} \\
    \midrule
    BM3 & mbnv2(multimodal) & 720 & 0.03688 & 0.02904 & -21.26\% \\
    CombiGCN & mbnv2(multimodal) & 280 & 0.04055 & 0.01186 & -70.75\% \\
    FREEDOM & mbnv2(mm\_attn) & 960 & 0.46834 & 0.46978 & +0.31\% \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Nhận xét & Đánh giá chuyên sâu:
1.  **CombiGCN hội tụ siêu nhanh nhưng quá khớp mạnh:**
    *   CombiGCN đạt đỉnh hiệu năng validation rất sớm ở **epoch 280**. Sau điểm này, Loss huấn luyện tiếp tục giảm sâu đến **70.75%** (từ 0.04055 xuống 0.01186) nhưng chỉ số gợi ý trên tập kiểm thử bắt đầu đi xuống.
    *   **Bản chất:** CombiGCN tích hợp trực tiếp các đặc trưng visual/text vào ma trận kề đồ thị tĩnh. Việc tối ưu hóa dựa trên liên kết này diễn ra rất nhanh nhưng cũng cực kỳ dễ bị quá khớp vào cấu trúc đồ thị huấn luyện khi chạy quá nhiều epoch mà không có các lớp regularization hoặc dropout mạnh mẽ.
2.  **BM3 hội tụ chậm hơn nhưng chống quá khớp xuất sắc:**
    *   BM3 cần đến **epoch 720** để đạt đỉnh hiệu năng. Sau đó, loss huấn luyện chỉ giảm nhẹ **21.26%** từ epoch 720 đến 1000.
    *   **Bản chất:** BM3 sử dụng cơ chế contrastive learning (học tương phản tự giám sát) đa góc nhìn. Quá trình căn chỉnh các không gian nhúng đa phương thức cần nhiều vòng lặp hơn để đạt trạng thái cân bằng. Tuy nhiên, chính mất mát tương phản tự giám sát đóng vai trò như một lớp regularization tự nhiên cực mạnh, giúp bảo toàn tính tổng quát hóa của không gian biểu diễn nhúng, ngăn chặn hiệu quả hiện tượng quá khớp ngay cả khi huấn luyện sâu.
3.  **FREEDOM hội tụ bất ổn định:**
    *   FREEDOM đạt `best_epoch` rất muộn ở **epoch 960**, đồng thời Loss huấn luyện biến động hầu như không đáng kể từ epoch 960 đến 1000 (+0.31%). Điều này cho thấy thuật toán học cấu trúc đồ thị phân tách của FREEDOM gặp khó khăn trong việc tìm hướng tối ưu hóa không gian biểu diễn nhúng, dẫn đến tiến trình hội tụ rất phẳng và chậm.

---

## 5. Kết Luận Và Đề Xuất Thiết Kế Hệ Thống Khuyến Nghị Thời Trang

Từ những phát hiện thực nghiệm đắt giá trên, chúng tôi đề xuất các hướng dẫn thiết kế sau cho Capstone Project:

1.  **Cấu hình khuyến nghị chính thức:** Triển khai mô hình **`BM3` kết hợp với visual encoder `MobileNetV2` và cơ chế liên kết Late Fusion (`multimodal`)**. Đây là cấu hình cho ra hiệu năng xếp hạng chiều sâu Top-5 và Top-10 tối ưu nhất trong môi trường dữ liệu thưa thớt thực tế.
2.  **Giải pháp thay thế ưu tiên tốc độ huấn luyện:** Nếu tài nguyên tính toán bị hạn chế hoặc cần cập nhật mô hình liên tục (real-time training), chọn **`CombiGCN` kết hợp với MobileNetV2 và Late Fusion**. CombiGCN cho kết quả xếp hạng rất tốt, đạt độ chính xác Top-1 ngang ngửa BM3 nhưng thời gian hội tụ nhanh hơn **2.5 lần** (epoch 280 so với 720).
3.  **Loại bỏ các cơ chế attention đơn giản:** Tránh cố gắng nhúng thêm các lớp attention học trọng số trực tiếp giữa visual và text trên tập dữ liệu tương tác thưa thớt để ngăn ngừa overfitting nghiêm trọng (như trường hợp của BM3 sụt giảm gần nửa hiệu năng).
