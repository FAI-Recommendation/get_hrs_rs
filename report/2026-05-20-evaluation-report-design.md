# Design Spec — Báo cáo tiến độ Evaluation (Chuyên đề 2)

**Ngày:** 2026-05-20
**Loại:** Báo cáo tiến độ (progress report), không phải báo cáo cuối kỳ
**Định dạng đầu ra:** File Markdown `.md`, ảnh chèn sau
**Mức độ phân tích:** Sâu (academic) — đặt Research Questions, trả lời bằng evidence từ chart
**Scope:** Chỉ phần Evaluation (24 configs, không bao gồm data pipeline / model architecture)
**Approach đã chọn:** C — Hybrid (RQ-driven + Funnel narrative)

---

## 1. Mục tiêu

Tạo một báo cáo tiến độ Markdown trình bày kết quả đánh giá 24 model configs của hệ
thống gợi ý thời trang. Báo cáo phải:

1. Thể hiện rõ **bạn đang hỏi gì** (Research Questions)
2. Có **đủ evidence** từ chart để trả lời từng câu hỏi
3. Phân tích có **lập luận nguyên nhân**, không chỉ mô tả kết quả
4. Chọn lọc **23/82 ảnh** — đủ để chứng minh, không gây nhiễu

### Bối cảnh dữ liệu (để báo cáo tự chứa ngữ cảnh)

| Chiều | Giá trị |
|---|---|
| Models | `combigcn`, `bm3`, `freedom` |
| Encoders | `clip`, `mbnv2` |
| Sim types | `img_only`, `tfidf`, `multimodal`, `multimodal_attention` |
| Metrics | `recall`, `precision`, `ndcg`, `hit_ratio`, `map`, `mrr` |
| K values | 1, 5, 10, 20 |
| Tổng configs | 24 = 3 models × 2 encoders × 4 sim_types |
| Dataset | 553 users, 2.194 items, ~3.8 test items/user |

---

## 2. Khung phân tích — 4 Research Questions

> **RQ1:** Encoder nào (CLIP vs MBNv2) phù hợp hơn cho domain thời trang?
>
> **RQ2:** Sim_type nào hiệu quả nhất? `multimodal_attention` có cải thiện so với
> `multimodal` thuần không?
>
> **RQ3:** Model nào (BM3 / CombiGCN / FREEDOM) tốt nhất với best config của nó?
>
> **RQ4:** Kết quả có nhất quán qua K và metric không? Kết luận tổng thể?

---

## 3. Cấu trúc báo cáo & lựa chọn chart

### Section 1 — Mở đầu: Khung 4 Research Questions
Đặt 4 RQ làm xương sống. Không có chart. Giải thích ngắn tại sao chọn 4 câu hỏi này.

### Section 2 — Tổng quan 24 configs (Context setting)

| Ảnh | Vai trò |
|---|---|
| `Figure_04_NDCG_K_Model_configs_heatmap_All_encoders.png` | Thấy **số cụ thể** của 24 configs |
| `Figure_07_NDCG_K_All_model_configs_line_All_encoders.png` | Thấy **phân tầng** rõ ràng, config nào vượt trội |

**Phân tích cần có:** phân bố chung, khoảng cách top-vs-bottom, metric nào phân
tách rõ nhất. Dẫn vào RQ1.

### Section 3 — RQ1: Encoder (CLIP vs MBNv2)

| Ảnh | Vai trò |
|---|---|
| `Figure_46_Recall_Precision_NDCG_K_CLIP.png` | 3 metrics chính của 12 config CLIP |
| `Figure_47_Recall_Precision_NDCG_K_MBNV2.png` | 3 metrics chính của 12 config MBNv2 |
| `Figure_74_Radar_Overview_CLIP_12_Configs.png` | Hình dạng tổng thể CLIP |
| `Figure_75_Radar_Overview_MBNv2_12_Configs.png` | Hình dạng tổng thể MBNv2 |

**Phân tích cần có:** so sánh số ★ thắng giữa 2 encoder, diện tích radar, encoder
nào ổn định hơn qua K. Kết luận RQ1.

### Section 4 — RQ2: Sim_type Ablation (3 models)

| Model | Ảnh | Vai trò |
|---|---|---|
| BM3 | `Figure_48_NDCG_K_BM3_ablation.png` | Heatmap NDCG: CLIP vs MBNv2 × sim_type |
| BM3 | `Figure_54_BM3_NDCG_5_by_sim_type_x_encoder.png` | Barplot NDCG@5 (phân tích thực) |
| BM3 | `Figure_55_BM3_NDCG_10_by_sim_type_x_encoder.png` | Barplot NDCG@10 (so literature) |
| CombiGCN | `Figure_56_NDCG_K_COMBIGCN_ablation.png` | Heatmap NDCG |
| CombiGCN | `Figure_62_COMBIGCN_NDCG_5_by_sim_type_x_encoder.png` | Barplot NDCG@5 |
| CombiGCN | `Figure_63_COMBIGCN_NDCG_10_by_sim_type_x_encoder.png` | Barplot NDCG@10 |
| FREEDOM | `Figure_64_NDCG_K_FREEDOM_ablation.png` | Heatmap NDCG |
| FREEDOM | `Figure_70_FREEDOM_NDCG_5_by_sim_type_x_encoder.png` | Barplot NDCG@5 |
| FREEDOM | `Figure_71_FREEDOM_NDCG_10_by_sim_type_x_encoder.png` | Barplot NDCG@10 |

**Phân tích cần có:** với từng model — sim_type nào thắng, CLIP hay MBNv2 nhạy hơn,
`multimodal_attention` có cải thiện so với `multimodal` không. Kết luận RQ2.

### Section 5 — RQ3: Best-vs-Best (Tier 1)

| Ảnh | Vai trò |
|---|---|
| `Figure_72_Tier_1_Best_Config_per_Model_Metrics_Overview.png` | Best config từng model trước khi so sánh |
| `Figure_76_Best-vs-Best_Model_Comparison.png` | Lineplot grid 6 metrics @ K=5 |
| `Figure_77_Best-vs-Best_NDCG_K.png` | NDCG@K phóng to @ K=5 |
| `Figure_78_Overall_Model_Performance_Radar.png` | Radar best-vs-best @ K=5 |
| `Figure_79_Best-vs-Best_Model_Comparison.png` | Lineplot grid 6 metrics @ K=10 |
| `Figure_80_Best-vs-Best_NDCG_K.png` | NDCG@K phóng to @ K=10 |
| `Figure_81_Overall_Model_Performance_Radar.png` | Radar best-vs-best @ K=10 |

**Phân tích cần có:** BM3 vs CombiGCN khoảng cách bao nhiêu %, FREEDOM kém ở đâu,
K=5 và K=10 có chọn cùng best config không. Kết luận RQ3.

### Section 6 — RQ4: Kết luận tổng thể

| Ảnh | Vai trò |
|---|---|
| `Figure_82_Best_Overall_Models_for_Each_Metric.png` | Tổng kết: config nào thắng metric nào |

**Phân tích cần có:** model config nào thắng nhiều metric nhất, metric nào biên độ
lớn nhất, tổng kết 4 RQ thành bảng kết luận.

---

## 4. Bảng tổng hợp lựa chọn ảnh

| Section | Ảnh | Số lượng |
|---|---|---|
| 1. Khung RQ | (không chart) | 0 |
| 2. Context | Fig 04, 07 | 2 |
| 3. RQ1 Encoder | Fig 46, 47, 74, 75 | 4 |
| 4. RQ2 Ablation | Fig 48, 54, 55, 56, 62, 63, 64, 70, 71 | 9 |
| 5. RQ3 Best-vs-Best | Fig 72, 76, 77, 78, 79, 80, 81 | 7 |
| 6. RQ4 Kết luận | Fig 82 | 1 |
| **Tổng** | | **23 / 82** |

---

## 5. Nguyên tắc lựa chọn ảnh (lý do loại bỏ 59 ảnh còn lại)

1. **Mỗi ảnh phải trả lời ít nhất 1 RQ cụ thể.** Ảnh không gắn được vào RQ nào → loại.
2. **Không giữ ảnh trùng thông tin.** Fig 01-45 là bộ khám phá đầy đủ (5 metrics × 3
   scopes × 3 chart types); chỉ cần 2 ảnh NDCG đại diện cho Context vì Recall/
   Precision/MAP được phân tích sâu hơn ở RQ2/RQ3.
3. **Ưu tiên ảnh thiết kế để so sánh trực tiếp** hơn ảnh nhìn đơn lẻ. Fig 46-47 đặt
   CLIP/MBNv2 cạnh nhau → giữ; histogram encoder-only rời rạc → loại.
4. **Heatmap ablation:** chỉ giữ NDCG (Fig 48/56/64), bỏ Recall/Precision/HIT_RATIO/
   MAP/MRR heatmap vì pattern tương tự, chỉ tốn diện tích.
5. **K=10 bắt buộc giữ song song K=5** ở RQ3 để so sánh với paper gốc (BM3,
   FREEDOM, CombiGCN đều dùng K=10).

---

## 6. Tiêu chí hoàn thành (success criteria)

- [ ] File `.md` có 6 section đúng cấu trúc trên
- [ ] 4 RQ được nêu rõ ở đầu và mỗi RQ có section trả lời tương ứng
- [ ] 23 placeholder ảnh được chèn đúng vị trí (đường dẫn tương đối tới `charts/`)
- [ ] Mỗi ảnh có caption + đoạn phân tích có lập luận (không chỉ mô tả)
- [ ] Mỗi RQ kết thúc bằng 1 câu kết luận rõ ràng
- [ ] Section 6 có bảng tổng kết 4 RQ
- [ ] Báo cáo tự chứa bối cảnh dữ liệu (người đọc không cần đọc tài liệu khác)
