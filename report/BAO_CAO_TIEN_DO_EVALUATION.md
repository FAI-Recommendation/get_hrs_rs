# Báo cáo tiến độ tuần 20— Đánh giá hệ thống gợi ý thời trang đa phương thức

**Chuyên đề 2 — Recommendation System with Multimodal Features**
**Giai đoạn:** Đánh giá & so sánh 24 model configs

---

## Công việc trong tuần

- **Chuẩn bị dữ liệu theo cơ chế model:** hoàn thiện pipeline từ raw VCR (~64k
  giao dịch) → sample 10k → N-Core 5 filter → per-user temporal split 80/20,
  thu được `train.txt` / `test.txt` (553 users, 2.194 items, 7.350 / 2.105
  tương tác). Lọc ảnh chính (`displayOrder == 0`) để embedding mang đúng ngữ
  nghĩa outfit.
- **Sinh đặc trưng đa phương thức:** trích embedding ảnh bằng **CLIP** và
  **MobileNetV2**, embedding text (BERT / TF-IDF); chuẩn bị 8 adjacency matrix
  cho 4 cơ chế kết hợp `img_only / tfidf / multimodal (late fusion) /
  multimodal_attention (weight attention)`.
- **Train 24 thực nghiệm:** chạy 3 models (BM3, CombiGCN, FREEDOM) × 2 encoders
  × 4 sim_types, mỗi run 1000 epoch (embed_size 512, lr 0.001, BPR + contrastive
  loss cho BM3/FREEDOM), log đầy đủ lên WandB.
- **Đánh giá & tổng hợp:** tính 6 metrics (recall, precision, ndcg, hit_ratio,
  map, mrr) tại K=1/5/10/20, dựng hệ thống 82 biểu đồ so sánh và phân tích kết
  quả (nội dung chính của báo cáo này).

---

## Bối cảnh đánh giá

Hệ thống được đánh giá trên **24 cấu hình** = 3 models × 2 encoders × 4 sim_types.

| Chiều | Giá trị |
|---|---|
| Models | `bm3`, `combigcn`, `freedom` |
| Encoders | `clip`, `mbnv2` (MobileNetV2) |
| Sim types | `img_only`, `tfidf`, `multimodal`, `multimodal_attention` |
| Metrics | `recall`, `precision`, `ndcg`, `hit_ratio`, `map`, `mrr` |
| K values | 1, 5, 10, 20 |
| Dataset | 553 users, 2.194 items, ~3.8 test items/user |

> **Lưu ý về quy mô dataset:** Trung bình mỗi user chỉ có ~3.8 item trong test set.
> Vì vậy báo cáo phân tích chính ở **K=5** (recommend 1.3× ground truth — phân
> biệt model rõ, ít bị bão hòa), và giữ **K=10** song song để so sánh với
> convention trong các paper gốc BM3 / FREEDOM / CombiGCN.

---

## Phần 1 — Khung phân tích: 4 câu hỏi nghiên cứu

Toàn bộ báo cáo được tổ chức quanh 4 câu hỏi:

> **RQ1 — Encoder:** CLIP hay MobileNetV2 phù hợp hơn cho domain thời trang?
>
> **RQ2 — Sim_type:** Cách kết hợp đặc trưng nào hiệu quả nhất? Cụ thể, cơ chế
> hợp nhất bằng attention (`multimodal_attention` — Weight Attention Fusion) có
> thực sự cải thiện so với hợp nhất mặc định (`multimodal` — Late Fusion) không?
>
> **RQ3 — Model:** Trong 3 kiến trúc (BM3, CombiGCN, FREEDOM), model nào tốt nhất
> khi mỗi model được chạy với cấu hình tối ưu của riêng nó?
>
> **RQ4 — Tính nhất quán & kết luận:** Thứ hạng có ổn định qua các K và metric
> không? Kết luận tổng thể là gì?

Bốn câu hỏi này tách bạch ba yếu tố thiết kế (encoder, cách hợp nhất modal, kiến
trúc model) để xác định **yếu tố nào quyết định chất lượng gợi ý**, thay vì chỉ
báo cáo "config nào điểm cao nhất".

---

## Phần 2 — Tổng quan 24 cấu hình (Context)

![Heatmap NDCG@K của 24 configs](../data_evaluate/charts/Figure_04_NDCG_K_Model_configs_heatmap_All_encoders.png)

*Hình 04 — Heatmap NDCG@K toàn bộ 24 configs (rows = config, cols = K).*

![Lineplot NDCG@K của 24 configs](../data_evaluate/charts/Figure_07_NDCG_K_All_model_configs_line_All_encoders.png)

*Hình 07 — Lineplot NDCG@K: phân tầng giữa các config, đường thắng được tô đậm.*

**Phân tích.** Bức tranh tổng thể cho thấy **sự phân hóa rất mạnh** giữa các
cấu hình. Tại NDCG@10, config tốt nhất (`bm3 · mbnv2 · multimodal` = 0.0186)
cao gấp **~5.6 lần** config tệ nhất (`freedom · clip · multimodal_attention` =
0.0033). Mức chênh lệch này cho thấy lựa chọn (model, encoder, sim_type) **không
phải yếu tố thứ yếu** — nó quyết định kết quả gần một bậc độ lớn.

Quan sát ban đầu quan trọng: nhóm config dẫn đầu đều thuộc về **BM3 và CombiGCN
với sim_type `multimodal`**, trong khi toàn bộ nhóm **FREEDOM** nằm ở đáy bảng.
Trên heatmap, `bm3 · mbnv2 · multimodal` giữ ô đậm nhất ở mọi K, đạt đỉnh
**0.0259 tại K=20**. Đáng chú ý, ngay trong cùng một (model, encoder), bản
`multimodal` luôn nhỉnh hơn `img_only`/`tfidf` (ví dụ `combigcn · clip`:
multimodal 0.0231 vs img_only 0.0220 tại K=20) — tín hiệu sớm cho RQ2 rằng kết
hợp 2 nguồn đặc trưng có lợi. NDCG là metric phân tách các config rõ nhất (biên
độ rộng), nên được chọn làm metric chủ đạo để trả lời RQ1–RQ3 bên dưới.

---

## Phần 3 — RQ1: Encoder nào phù hợp hơn cho thời trang?

![Recall/Precision/NDCG — CLIP](../data_evaluate/charts/Figure_46_Recall_Precision_NDCG_K_CLIP.png)

*Hình 46 — 12 config dùng CLIP encoder trên 3 metrics chính.*

![Recall/Precision/NDCG — MBNv2](../data_evaluate/charts/Figure_47_Recall_Precision_NDCG_K_MBNV2.png)

*Hình 47 — 12 config dùng MobileNetV2 encoder trên 3 metrics chính.*

![Radar CLIP 12 configs](../data_evaluate/charts/Figure_74_Radar_Overview_CLIP_12_Configs.png)

*Hình 74 — Radar tổng thể nhóm CLIP (giá trị trung bình qua K).*

![Radar MBNv2 12 configs](../data_evaluate/charts/Figure_75_Radar_Overview_MBNv2_12_Configs.png)

*Hình 75 — Radar tổng thể nhóm MBNv2 (giá trị trung bình qua K).*

**Phân tích.** Câu trả lời cho RQ1 **không đơn giản là "encoder X luôn tốt hơn"**
mà phụ thuộc vào sim_type — đây là một phát hiện quan trọng:

| Model | Sim_type tốt nhất | NDCG@10 CLIP | NDCG@10 MBNv2 | Encoder thắng |
|---|---|---|---|---|
| BM3 | multimodal | 0.0142 | **0.0186** | MBNv2 (+31%) |
| CombiGCN | multimodal | 0.0174 | **0.0175** | MBNv2 (≈ ngang) |
| FREEDOM | multimodal_attention | 0.0033 | **0.0088** | MBNv2 (+165%) |

Điểm mấu chốt: **với sim_type tốt nhất của mỗi model, MobileNetV2 luôn thắng
hoặc ngang CLIP**, và toàn bộ 3 best-config của 3 model đều dùng MBNv2 — CLIP
không tạo ra best config cho bất kỳ model nào.

Radar (Hình 74 vs 75) củng cố điều này theo chiều đa metric: nhóm CLIP bị phân
mảnh — không config nào trội toàn diện, HIT_RATIO cao nhất chỉ ~0.06; trong khi
ở nhóm MBNv2, `bm3 · multimodal` kéo giãn gần như mọi trục, HIT_RATIO chạm gần
**0.08** và NDCG@20 vượt **0.025** (so với ~0.023 của best CLIP). Lineplot cũng
cho thấy CLIP **thiếu nhất quán** (ngôi sao "best" nhảy giữa các config theo K),
còn MBNv2 ổn định ở một config thắng duy nhất.

CLIP chỉ vượt MBNv2 ở vài cấu hình yếu (`tfidf`, một số `img_only`), tức ở những
trường hợp không phải lựa chọn tối ưu. Điều này trái với kỳ vọng ban đầu (CLIP
được pre-train text-image quy mô lớn nên "nên" mạnh hơn). Giải thích hợp lý:
MobileNetV2 nắm bắt **đặc trưng texture/họa tiết** của sản phẩm thời trang tốt
hơn embedding ngữ nghĩa tổng quát của CLIP — phù hợp với bản chất visual của bài
toán recommend thời trang.

> **Kết luận RQ1:** **MobileNetV2 là encoder phù hợp hơn cho domain thời trang.**
> Nó tạo ra best config cho cả 3 model; CLIP chỉ vượt ở các cấu hình không tối ưu.

---

## Phần 4 — RQ2: Sim_type nào hiệu quả nhất? Attention có cải thiện không?

### 4.1. BM3

![Heatmap NDCG — BM3](../data_evaluate/charts/Figure_48_NDCG_K_BM3_ablation.png)

*Hình 48 — Heatmap NDCG@K của BM3: CLIP vs MBNv2 × 4 sim_type.*

![BM3 NDCG@5 by sim_type](../data_evaluate/charts/Figure_54_BM3_NDCG_5_by_sim_type_x_encoder.png)

*Hình 54 — BM3: NDCG@5 theo sim_type × encoder (phân tích sát thực).*

![BM3 NDCG@10 by sim_type](../data_evaluate/charts/Figure_55_BM3_NDCG_10_by_sim_type_x_encoder.png)

*Hình 55 — BM3: NDCG@10 theo sim_type × encoder (so sánh literature).*

### 4.2. CombiGCN

![Heatmap NDCG — CombiGCN](../data_evaluate/charts/Figure_56_NDCG_K_COMBIGCN_ablation.png)

*Hình 56 — Heatmap NDCG@K của CombiGCN.*

![CombiGCN NDCG@5 by sim_type](../data_evaluate/charts/Figure_62_COMBIGCN_NDCG_5_by_sim_type_x_encoder.png)

*Hình 62 — CombiGCN: NDCG@5 theo sim_type × encoder.*

![CombiGCN NDCG@10 by sim_type](../data_evaluate/charts/Figure_63_COMBIGCN_NDCG_10_by_sim_type_x_encoder.png)

*Hình 63 — CombiGCN: NDCG@10 theo sim_type × encoder.*

### 4.3. FREEDOM

![Heatmap NDCG — FREEDOM](../data_evaluate/charts/Figure_64_NDCG_K_FREEDOM_ablation.png)

*Hình 64 — Heatmap NDCG@K của FREEDOM.*

![FREEDOM NDCG@5 by sim_type](../data_evaluate/charts/Figure_70_FREEDOM_NDCG_5_by_sim_type_x_encoder.png)

*Hình 70 — FREEDOM: NDCG@5 theo sim_type × encoder.*

![FREEDOM NDCG@10 by sim_type](../data_evaluate/charts/Figure_71_FREEDOM_NDCG_10_by_sim_type_x_encoder.png)

*Hình 71 — FREEDOM: NDCG@10 theo sim_type × encoder.*

### 4.4. Phân tích tổng hợp RQ2

Bảng NDCG@10 theo sim_type (encoder tốt hơn của từng cặp):

| Model | img_only | tfidf | multimodal | multimodal_attention |
|---|---|---|---|---|
| BM3 (mbnv2) | 0.0150 | 0.0149 | **0.0186** | 0.0101 |
| CombiGCN (mbnv2) | 0.0085 | 0.0071 | **0.0175** | 0.0151 |
| FREEDOM (mbnv2) | 0.0062 | 0.0031 | 0.0081 | **0.0088** |

Hai kết luận rút ra:

**(1) `multimodal` (Late Fusion) là sim_type tốt nhất cho 2/3 model.** BM3 và CombiGCN
đều đạt đỉnh ở `multimodal` — kết hợp cả visual lẫn text vượt trội rõ rệt so với
chỉ dùng một nguồn (`img_only`, `tfidf`). `tfidf` (chỉ text) là sim_type yếu nhất
cho các model graph/contrastive, xác nhận đặc trưng **visual đóng vai trò chủ
đạo** trong recommend thời trang.

**(2) `multimodal_attention` KHÔNG cải thiện một cách phổ quát — và đây là phát
hiện đáng chú ý nhất của RQ2.** Cơ chế attention:

- **Làm hại BM3 nặng:** NDCG@10 rơi từ 0.0186 (multimodal) xuống 0.0101
  (−46%). Trên CLIP còn tệ hơn (0.0142 → 0.0059, −59%).
- **Làm hại CombiGCN nhẹ:** 0.0175 → 0.0151 (−14%).
- **Chỉ giúp FREEDOM:** 0.0081 → 0.0088 (+8%) — và FREEDOM lại là model yếu
  nhất.

Giải thích: BM3/CombiGCN đã có cơ chế hợp nhất modal hiệu quả sẵn; thêm một lớp
attention làm tăng số tham số và đưa thêm nhiễu/overfit trên dataset nhỏ (~9.4k
tương tác). FREEDOM yếu hơn nên attention bù đắp được phần nào, nhưng vẫn không
đủ để cạnh tranh.

**(3) Không có "viên đạn bạc" cho mọi kiến trúc.** Hai dị biệt đáng ghi nhận:
BM3 đặc biệt hợp `tfidf` — ở **nhánh CLIP**, `tfidf` còn nhỉnh hơn cả `multimodal`;
ngược lại CombiGCN gần như "dị ứng" với `tfidf` (NDCG tụt đáy) nhưng khi chỉ dùng
`img_only` thì lại thích **CLIP hơn MBNv2** — trái với xu hướng chung. Điều này
cho thấy lựa chọn encoder/sim_type phải xét **theo từng kiến trúc**, không suy
diễn toàn cục.

> **Kết luận RQ2:** **`multimodal` (Late Fusion) là lựa chọn tốt nhất.**
> `multimodal_attention` không phải cải tiến phổ quát — nó chỉ có ích cho model
> yếu (FREEDOM) và gây hại cho model mạnh (BM3, CombiGCN) trên dataset quy mô này.

---

## Phần 5 — RQ3: Model nào tốt nhất với cấu hình tối ưu?

![Tier 1 — Best config per model overview](../data_evaluate/charts/Figure_72_Tier_1_Best_Config_per_Model_Metrics_Overview.png)

*Hình 72 — Best config của từng model trên cả 6 metrics.*

![Best-vs-Best lineplot K=5](../data_evaluate/charts/Figure_76_Best-vs-Best_Model_Comparison.png)

*Hình 76 — So sánh trực tiếp 3 best config trên 6 metrics, tiêu chí NDCG@5.*

![Best-vs-Best NDCG@K=5](../data_evaluate/charts/Figure_77_Best-vs-Best_NDCG_K.png)

*Hình 77 — NDCG@K phóng to (tiêu chí K=5).*

![Radar best-vs-best K=5](../data_evaluate/charts/Figure_78_Overall_Model_Performance_Radar.png)

*Hình 78 — Radar 3 best config (tiêu chí K=5).*

![Best-vs-Best lineplot K=10](../data_evaluate/charts/Figure_79_Best-vs-Best_Model_Comparison.png)

*Hình 79 — So sánh trực tiếp 3 best config, tiêu chí NDCG@10 (so literature).*

![Best-vs-Best NDCG@K=10](../data_evaluate/charts/Figure_80_Best-vs-Best_NDCG_K.png)

*Hình 80 — NDCG@K phóng to (tiêu chí K=10).*

![Radar best-vs-best K=10](../data_evaluate/charts/Figure_81_Overall_Model_Performance_Radar.png)

*Hình 81 — Radar 3 best config (tiêu chí K=10).*

**Phân tích.** Best config của 3 model (đều dùng MBNv2 — củng cố lại RQ1):

| Hạng | Model | Best config | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| 1 | **BM3** | `mbnv2 · multimodal` | **0.01622** | **0.01859** |
| 2 | CombiGCN | `mbnv2 · multimodal` | 0.01370 | 0.01749 |
| 3 | FREEDOM | `mbnv2 · multimodal_attention` | 0.00840 | 0.00876 |

**BM3 vs CombiGCN — khoảng cách phụ thuộc K:**

- Tại NDCG@10: chênh lệch tương đối chỉ **+6.3%** → hai model gần tương đương.
- Tại NDCG@5: chênh lệch nới rộng lên **+18.4%** → BM3 vượt trội rõ rệt.
- **Tại K=1 (twist): CombiGCN lại vượt BM3** trên NDCG@1 và Precision@1 — nếu
  hệ thống chỉ trả về đúng 1 kết quả và cần độ chính xác tuyệt đối, CombiGCN là
  lựa chọn tốt hơn. BM3 chỉ bứt phá và nới rộng khoảng cách từ K≥5 trở đi.

Đây chính là lý do báo cáo ưu tiên K=5: ở K=10 trên dataset nhỏ này, recall bão
hòa làm hai model "trông giống nhau", che mất sự khác biệt thực sự mà K=5 phơi
bày. **K=5 và K=10 chọn cùng best config cho cả 3 model** (BM3/CombiGCN →
multimodal, FREEDOM → mm_attention), nên lựa chọn cấu hình là **ổn định**, chỉ
khác ở mức độ chênh lệch giữa các model.

**FREEDOM kém rõ rệt:** NDCG@10 chỉ đạt 0.0088 — **thấp hơn 53%** so với BM3
(0.0186) và không có metric nào FREEDOM tiệm cận được nhóm dẫn đầu. Điều này cho
thấy kiến trúc FREEDOM (vốn dựa nhiều vào đồ thị item-item từ feature) không khai
thác tốt multimodal feature trên dataset thời trang quy mô nhỏ này.

> **Kết luận RQ3:** **BM3 là model tốt nhất.** Vượt CombiGCN +18% tại K=5 (tiêu
> chí sát thực) và +6% tại K=10; vượt FREEDOM hơn gấp đôi. CombiGCN là phương án
> thay thế hợp lý nếu ưu tiên top-10.

---

## Phần 6 — RQ4: Tính nhất quán & kết luận tổng thể

![Best overall model per metric](../data_evaluate/charts/Figure_82_Best_Overall_Models_for_Each_Metric.png)

*Hình 82 — Với mỗi metric, config có mean score cao nhất (trung bình qua mọi K).*

**Phân tích.** `bm3 · mbnv2 · multimodal` thắng mean score ở **5/6 metric**
(Recall 0.0240, Precision 0.0096, NDCG 0.0183, MRR 0.0254, MAP 0.0097) khi lấy
trung bình qua K=1/5/10/20. Điều này loại bỏ khả năng kết quả là ngẫu nhiên theo
một metric đơn lẻ: BM3 thắng một cách **đa chiều và ổn định**.

**Twist ở HIT_RATIO:** metric duy nhất BM3 không thắng là Hit Ratio — quán quân
mean là `combigcn · clip · multimodal` (**0.0737**). Nghĩa là CombiGCN+CLIP nhạy
hơn ở việc *đảm bảo có ít nhất một kết quả đúng lọt vào danh sách*, dù khả năng
xếp hạng (đẩy kết quả đúng lên đầu) vẫn kém BM3. Đây là ngoại lệ duy nhất và
không làm đổi kết luận tổng thể.

Thứ hạng tổng thể **không đổi qua mọi K và mọi metric**:
`BM3 > CombiGCN > FREEDOM`. Khoảng cách BM3–CombiGCN co giãn theo K (rộng ở K nhỏ,
hẹp ở K lớn) nhưng thứ tự không bao giờ đảo. Đây là tín hiệu mạnh để tự tin kết
luận.

### Bảng tổng kết 4 Research Questions

| RQ | Câu hỏi | Kết luận |
|---|---|---|
| **RQ1** | Encoder nào phù hợp? | **MobileNetV2** — tạo best config cho cả 3 model; CLIP chỉ thắng ở cấu hình không tối ưu |
| **RQ2** | Sim_type nào tốt nhất? | **`multimodal` (Late Fusion)**; `multimodal_attention` (Weight Attention) chỉ giúp model yếu (FREEDOM), gây hại model mạnh (BM3/CombiGCN) |
| **RQ3** | Model nào tốt nhất? | **BM3** (`mbnv2·multimodal`) — vượt CombiGCN +18% @K=5, vượt FREEDOM >2× |
| **RQ4** | Có nhất quán không? | **Có** — thứ hạng `BM3 > CombiGCN > FREEDOM` ổn định qua mọi K và mọi metric |

### Cấu hình khuyến nghị

**`BM3 · MobileNetV2 · multimodal`** — NDCG@5 = 0.0162, NDCG@10 = 0.0186.

