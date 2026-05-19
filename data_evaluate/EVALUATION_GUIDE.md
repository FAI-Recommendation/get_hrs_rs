# Evaluation Guide — So sánh Models cho Chuyên đề 2

Notebook `evaluate.ipynb` đánh giá và so sánh hiệu năng của các mô hình gợi ý (recommendation) dựa trên dữ liệu từ WandB. Phân tích đi từ **tổng quan** → **phân tách theo encoder** → **ablation từng model** → **radar landscape** → **so sánh trực tiếp best-vs-best** → **best overall**.

---

## Dữ liệu đầu vào

```python
df = load_from_csv("data_wandb/all_runs_summary.csv")
```

**24 runs** tổng cộng, mỗi run là một tổ hợp `(model, encoder, sim_type)`.

| Chiều | Giá trị |
|---|---|
| **Models** | `combigcn`, `bm3`, `freedom` |
| **Encoders** | `clip`, `mbnv2` |
| **Sim types** | `img_only`, `tfidf`, `multimodal`, `multimodal_attention` |
| **Metrics** | `recall`, `precision`, `ndcg`, `hit_ratio`, `map`, `mrr` |
| **K values** | 1, 5, 10, 20 |

> **Q&A — Trước khi bắt đầu**
>
> **Q: Encoder nào (CLIP hay MBNv2) được kỳ vọng mạnh hơn trên domain thời trang?**  
> → CLIP được pre-train trên text-image pairs quy mô lớn nên có thể nắm bắt ngữ nghĩa visual tốt hơn. Tuy nhiên MBNv2 là backbone nhẹ, có thể khớp tốt hơn với đặc trưng texture của sản phẩm thời trang. Xem kết quả ở phần Encoder Breakdown.
>
> **Q: sim_type nào dự kiến tốt nhất?**  
> → `multimodal` kết hợp cả visual lẫn text nên thường dẫn đầu. `tfidf` chỉ dựa vào text, `img_only` chỉ visual. `multimodal_attention` là biến thể có cơ chế attention — thực nghiệm sẽ cho biết nó có cải thiện không.
>
> **Q: 3 models có điểm xuất phát (kiến trúc) khác nhau thế nào?**  
> → BM3 và FREEDOM là các model contrastive learning mới; CombiGCN dựa trên graph convolution với side information. Kỳ vọng BM3/CombiGCN vượt FREEDOM trong setting có multimodal features.

---

## Histogram & Heatmap — Tổng quan tất cả model configs

```python
plot_overview_all_metrics(df)
```

So sánh toàn bộ 24 model configs trên **5 metrics** (`ndcg`, `recall`, `precision`, `map`, `mrr`).  
Mỗi metric được vẽ theo 3 phạm vi:

| Phạm vi | Số models | Hàm |
|---|---|---|
| Full (cả 2 encoder) | 24 | `plot_histogram_models(df)` |
| CLIP only | 12 | `plot_histogram_models(df, encoder='clip')` |
| MBNv2 only | 12 | `plot_histogram_models(df, encoder='mbnv2')` |

**3 loại chart cho mỗi phạm vi:**

- **Histogram (bar chart dọc):** Trục X = model config, trục Y = metric, grouped bars = K=1/5/10/20. Sort từ tốt nhất sang kém nhất.
- **Heatmap:** Rows = model configs, cols = K. Giá trị annotate trực tiếp (`fmt=".4f"`).
- **Lineplot:** Đường thắng tại ≥1 K được highlight (`linewidth=2.8, alpha=1.0`), còn lại mờ (`alpha=0.35`). Dấu ★ đỏ + tên model tại điểm tốt nhất mỗi K.

**Vẽ riêng lẻ:**
```python
plot_histogram_models(df, encoder='clip', metric='recall')
plot_heatmap_models(df, encoder='mbnv2', metric='ndcg')
plot_lineplot_models(df, metric='map')
```

> **Q&A — Sau tổng quan 24 configs**
>
> **Q: Khoảng cách giữa top-1 và top-10 config lớn không?**  
> → Nếu lớn: có sự phân hóa rõ rệt giữa các lựa chọn encoder/sim_type. Nếu nhỏ: cần zoom vào từng model để thấy sự khác biệt → xem Encoder Breakdown bên dưới.
>
> **Q: Metric nào phân tách các configs rõ nhất?**  
> → NDCG và MRR thường có biên độ lớn hơn Precision. Nếu Recall phân tách tốt: model quan tâm đến độ phủ. Nếu Precision phân tách tốt: model ưu tiên chất lượng top-K.
>
> **Q: Config tệ nhất (bên phải histogram) tệ hơn config tốt nhất bao nhiêu lần?**  
> → Tỉ lệ này cho biết mức độ "nhạy cảm" của kết quả với lựa chọn hyperparameter. → Sang phần Encoder Breakdown để so sánh CLIP vs MBNv2 trên 3 metrics chính.

---

## Encoder Breakdown — Recall / Precision / NDCG theo từng Encoder

```python
plot_lineplot_3metrics_both(df)
```

Sau khi thấy bức tranh tổng thể 24 configs, phần này **tách riêng CLIP và MBNv2** để so sánh trực tiếp trong cùng metric space.

**2 figures riêng biệt, mỗi figure có 1×3 subplots:**

| Subplot | Metric |
|---|---|
| 1 | Recall@K |
| 2 | Precision@K |
| 3 | NDCG@K |

Mỗi subplot vẽ **12 model configs** (cho encoder đó). Legend chung bên phải figure.  
Đường thắng tại ≥1 K được highlight đậm. Dấu ★ đỏ + annotation tên tại điểm tốt nhất mỗi K.

**Vẽ riêng hoặc đổi metrics:**
```python
plot_lineplot_3metrics(df, encoder='clip')
plot_lineplot_3metrics(df, encoder='mbnv2')
plot_lineplot_3metrics(df, encoder='clip', metrics=['ndcg', 'map', 'mrr'])
```

> **Q&A — Sau Encoder Breakdown**
>
> **Q: Encoder nào có nhiều ★ hơn (thắng nhiều K hơn)?**  
> → Đếm số ★ đỏ trên mỗi figure. Encoder thắng nhiều K hơn là lựa chọn ổn định hơn, không chỉ tốt tại một K cụ thể.
>
> **Q: Model config nào nhất quán dẫn đầu qua cả 3 metrics (Recall, Precision, NDCG)?**  
> → Nếu một config có ★ trên cả 3 subplots: đó là ứng viên mạnh cho Tầng 1. Nếu ★ phân tán: các metric ưu tiên các cấu hình khác nhau.
>
> **Q: sim_type nào thường xuất hiện ở top tại cả 2 encoder?**  
> → Nếu `multimodal` dẫn đầu ở cả CLIP và MBNv2: kết hợp modal là yếu tố quyết định, không phải encoder. → Sang Tầng 2 Ablation để phân tích từng model riêng lẻ.

---

## Tầng 2 — Ablation: Mỗi model làm gì tốt nhất?

```python
plot_ablation(df)
```

Phân tích ảnh hưởng của `encoder` (CLIP vs MobileNetV2) và `sim_type` lên từng model riêng lẻ.  
Chạy tuần tự cho `combigcn`, `bm3`, `freedom`.

**Với mỗi model:**

1. **Heatmap per model** (`plot_heatmap_per_model`):  
   2 heatmaps cạnh nhau (CLIP | MBNv2). Rows = sim_type, cols = K@1/5/10/20.  
   Chạy cho 6 metrics: recall, precision, ndcg, hit_ratio, map, mrr.

2. **Grouped barplot** (`plot_barplot_per_model`):  
   X = sim_type, grouped bars = encoder. Dấu ★ đỏ trên bar tốt nhất mỗi sim_type.  
   Vẽ **2 barplot**: NDCG@5 (phân tích thực) và NDCG@10 (so sánh literature).

3. **Ranking table** (`display_ranking_table`):  
   DataFrame styled với highlight ô max, sort theo NDCG@5 và NDCG@10 giảm dần.

> **Q&A — Sau Ablation từng model**
>
> **Q: Model nào bị ảnh hưởng nhiều nhất bởi lựa chọn encoder?**  
> → So sánh khoảng cách CLIP vs MBNv2 trên heatmap. Nếu model A cho kết quả gần nhau ở cả 2 encoder: model đó ổn định hơn với feature extraction. Nếu chênh lệch lớn: encoder là yếu tố quyết định.
>
> **Q: sim_type nào tệ nhất cho mỗi model và tại sao?**  
> → Thường là `img_only` hoặc `tfidf` vì chúng chỉ dùng 1 nguồn thông tin. Nếu `multimodal_attention` tệ hơn `multimodal`: cơ chế attention chưa mang lại lợi ích trên dataset này.
>
> **Q: FREEDOM có cấu hình nào vượt được BM3 hoặc CombiGCN không?**  
> → Xem ranking table — nếu không có: FREEDOM có vấn đề về kiến trúc hoặc hyperparameter cho task này. → Sang Tier 1 Preview để xem best config mỗi model.

---

## Tier 1 Preview — Metrics Histogram (Best config per model)

```python
plot_histogram_tier1(df)
```

Bước cầu nối giữa Tầng 2 và Tầng 1. Lấy **best config per model** (theo NDCG@10) rồi vẽ tổng quan trước khi so sánh trực tiếp.

**Grid 2×3** với 6 subplots, mỗi subplot = một metric:  
`recall`, `precision`, `ndcg`, `hit_ratio`, `map`, `mrr`

Mỗi subplot: bar chart dọc, X = best-config label của từng model, grouped bars = K=1/5/10/20.

> **Q&A — Sau Tier 1 Preview**
>
> **Q: Model nào có best config nhất quán dẫn đầu qua tất cả 6 metrics?**  
> → Nếu một model dẫn đầu ở cả 6 subplots: đây là model tổng thể tốt nhất, không chỉ tốt theo NDCG. Nếu mỗi metric có winner khác nhau: cần xác định metric nào phù hợp nhất với mục tiêu business.
>
> **Q: Metric nào có khoảng cách lớn nhất giữa model tốt nhất và tệ nhất?**  
> → Metric có biên độ lớn là metric "phân biệt" nhất — dùng làm tiêu chí chọn model. Metric nào các bar gần bằng nhau: không đủ sức phân biệt.
>
> **Q: Giữa K=1 và K=20, xu hướng thứ hạng model có thay đổi không?**  
> → Nếu thay đổi: model tốt ở top-1 chưa chắc tốt ở top-20 (và ngược lại). Cần xác định use case: precision (K nhỏ) hay recall (K lớn). → Sang Radar Overview để xem toàn cảnh trước Best-vs-Best.

---

## Radar Overview — Tổng quan tất cả configs

```python
plot_radar_overview(df)
```

Trước khi thu hẹp về 3 best configs, phần này cho thấy **toàn cảnh "hình dạng" performance** của mọi model config trên 6 metrics.

**3 radar charts liên tiếp:**

| Chart | Configs | Mô tả |
|---|---|---|
| **Radar — All 24 Configs** | 24 | Toàn bộ landscape: mỗi đường = 1 config, giá trị = mean qua K |
| **Radar — CLIP (12 Configs)** | 12 | Chỉ các config dùng CLIP encoder |
| **Radar — MBNv2 (12 Configs)** | 12 | Chỉ các config dùng MBNv2 encoder |

Mỗi radar: trục = 6 metrics (RECALL, PRECISION, NDCG, HIT\_RATIO, MAP, MRR), giá trị trung bình qua K=1/5/10/20.  
Diện tích lớn hơn = model config mạnh hơn tổng thể.

**Vẽ riêng từng encoder hoặc subset:**
```python
plot_radar_overview(df, save_dir="charts/radar")
# Vẽ 1 radar custom với bất kỳ subset:
from plot_tier2 import _plot_single_radar
from load_data import to_results_dict
results = to_results_dict(df[df["model"] == "bm3"])
_plot_single_radar(results, title="BM3 — All configs")
```

> **Q&A — Sau Radar Overview**
>
> **Q: Hình dạng radar của CLIP và MBNv2 có khác nhau nhiều không?**  
> → Nếu MBNv2 có diện tích lớn hơn ở NDCG/Recall: encoder thực sự tạo ra sự khác biệt. Nếu 2 nhóm chồng lên nhau: sim_type mới là yếu tố quyết định.
>
> **Q: Config nào có diện tích radar lớn nhất và đều nhất?**  
> → Config có diện tích lớn + đều (không quá lệch về 1 metric) là ứng viên mạnh nhất tổng thể. → Sang Best-vs-Best để so sánh chính thức 3 best configs.

---

## Tầng 1 — Best-vs-Best: Model nào tốt nhất?

```python
# Chạy 2 lần: K=5 (phân tích thực) và K=10 (so sánh literature)
best_df5, best_results5 = plot_best_vs_best(df, rank_metric="ndcg@5")
display(display_summary_table(best_df5))

best_df, best_results = plot_best_vs_best(df, rank_metric="ndcg@10")
display(display_summary_table(best_df))
```

> **Tại sao chạy cả K=5 và K=10?**
>
> Dataset này có ~3.8 test items/user trung bình (2.105 interactions / 553 users).  
> - **K=5** recommend 1.3× ground truth size → phân biệt model rõ hơn, ít bị inflate hơn.  
> - **K=10** recommend 2.6× ground truth → recall dễ bão hòa, nhưng là convention trong tất cả paper gốc (BM3, FREEDOM, CombiGCN) nên cần giữ để so sánh.

Lấy config tốt nhất của mỗi model theo từng tiêu chí rồi so sánh trực tiếp.

**Best configs (kết quả thực tế sau khi chạy notebook):**

| Tiêu chí | Model | Best config | Score |
|---|---|---|---|
| NDCG@5 | bm3 | _(chạy notebook để xem)_ | — |
| NDCG@5 | combigcn | _(chạy notebook để xem)_ | — |
| NDCG@5 | freedom | _(chạy notebook để xem)_ | — |
| NDCG@10 | bm3 | `mbnv2(multimodal)` | 0.018595 |
| NDCG@10 | combigcn | `mbnv2(multimodal)` | 0.017486 |
| NDCG@10 | freedom | `mbnv2(multimodal_attention)` | 0.008759 |

**3 loại chart** (chạy cho cả K=5 và K=10, radar best-vs-best luôn là chart cuối cùng):

1. **Lineplot grid** (`plot_lineplot_grid`):  
   Grid **2×3**, 6 metrics. Mỗi subplot: X = K, Y = metric, mỗi đường = một model.  
   Đường thắng highlight, ★ đỏ + annotation tên tại điểm tốt nhất mỗi K.

2. **Single lineplot** (`plot_lineplot_single`):  
   Phóng to chart NDCG@K riêng lẻ — rõ hơn khi cần trình bày.

3. **Radar chart — Best-vs-Best** (`plot_radar_chart`):  
   So sánh 3 best configs trên 6 metrics (trung bình qua K). Đây là radar **cuối cùng**, sau 3 radar overview ở phần trên.

**Summary table:**  
Styled DataFrame với highlight ô max per column. Hiển thị K=5, K=10, K=20. Rank 1 = model tốt nhất.

> **Q&A — Sau Best-vs-Best**
>
> **Q: Kết quả K=5 và K=10 có chọn cùng "best config" cho mỗi model không?**  
> → Nếu giống nhau: config tốt ổn định qua các K, tự tin lựa chọn. Nếu khác nhau: cần xem xét use case — ưu tiên phân biệt thực (K=5) hay so sánh với literature (K=10).
>
> **Q: Khoảng cách giữa BM3 (rank 1) và CombiGCN (rank 2) có đáng kể không?**  
> → Nếu < 5% relative difference: 2 model gần tương đương, lựa chọn có thể dựa trên tốc độ inference hoặc tính ổn định training. Nếu > 10%: BM3 rõ ràng vượt trội.
>
> **Q: FREEDOM kém BM3 bao nhiêu? Có metric nào FREEDOM không quá tệ không?**  
> → Nếu FREEDOM gần hơn trên Precision@1: nó có thể phù hợp hơn cho use case cần top-1 chính xác. Radar chart sẽ cho thấy "hình dạng" strength/weakness của mỗi model.
>
> **Q: Thứ hạng model có nhất quán qua K=1, 5, 10, 20 không?**  
> → Nếu model A thắng K=1 nhưng model B thắng K=20: cần chọn K phù hợp với product (gợi ý 1 item vs top-20). → Xem Best Overall bên dưới để kết luận tổng thể.

---

## Best Overall Model for Each Metric

```python
plot_best_overall_per_metric(df)
```

Chart kết luận của toàn bộ phân tích. Với **mỗi metric**, tìm model config có **mean score cao nhất** trung bình qua tất cả K=1/5/10/20.

**Bar chart:**
- Trục X = 6 metrics: RECALL, PRECISION, NDCG, HIT\_RATIO, MRR, MAP
- Trục Y = Mean Score (trung bình qua K)
- Mỗi bar = metric có màu riêng biệt
- Annotation trên mỗi bar: tên model config + giá trị (4 chữ số thập phân)

**Đọc chart:** Bar cao nhất cho biết metric nào dễ đạt được; config được annotate cho biết model nào thống trị metric đó sau khi trung bình hoá qua mọi K.

**Vẽ với subset hoặc metrics tùy chọn:**
```python
plot_best_overall_per_metric(df, metrics=["recall", "ndcg", "mrr"])
plot_best_overall_per_metric(df[df["encoder"] == "clip"])  # CLIP only
```

> **Q&A — Sau Best Overall**
>
> **Q: Có một model config duy nhất thắng trên tất cả 6 metrics không?**  
> → Nếu có: đó là config tốt nhất tuyệt đối. Nếu khác nhau: mỗi metric ưu tiên config khác nhau — xem xét metric phù hợp với use case.
>
> **Q: Metric nào có mean score cao nhất và thấp nhất?**  
> → Bar cao nhất = metric dễ đạt (model học được tốt). Bar thấp nhất = metric khó — Precision@K thường thấp nhất vì đòi hỏi chính xác ở từng vị trí. MRR cao nếu model đặt relevant item ở top-1 thường xuyên.

---

## Custom: Vẽ riêng từng chart

Cell cuối notebook chứa ví dụ để vẽ/save từng loại chart riêng lẻ khi cần customize chi tiết:

```python
# Heatmap một model + metric cụ thể
plot_heatmap_per_model(df, model="combigcn", metric="recall")

# Single lineplot NDCG phóng to
plot_lineplot_single(best_results, metric="ndcg")

# Radar overview với custom save dir
plot_radar_overview(df, save_dir="charts/radar")

# Best overall cho subset encoder
plot_best_overall_per_metric(df[df["encoder"] == "mbnv2"])

# Save toàn bộ ra folder
plot_ablation(df, save_dir="charts/tier2")
plot_best_vs_best(df, save_dir="charts/tier1")
```

> **Q&A — Hướng phân tích tiếp theo**
>
> **Q: Nếu thêm model mới (VD: SGL, SimGCL), cần làm gì?**  
> → Thêm runs vào CSV, gọi lại `load_from_csv()`. Tất cả hàm plot sẽ tự động include model mới — không cần sửa code plot.
>
> **Q: Nếu muốn so sánh theo metric khác thay vì NDCG làm rank?**  
> → Truyền `rank_metric="recall@5"` vào `plot_best_vs_best(df, rank_metric="recall@5")`. Best config được chọn sẽ thay đổi theo metric này.
>
> **Q: Muốn export bảng kết quả ra LaTeX/CSV cho báo cáo?**  
> → `summary_table(best_df).to_latex()` hoặc `.to_csv("results.csv")`. Hàm `summary_table` trả về DataFrame thuần, không phụ thuộc matplotlib.

---

## Cấu trúc file

```
data_evaluate/
├── evaluate.ipynb          # Notebook chính (27 cells)
├── load_data.py            # Load & transform data từ WandB CSV
├── plot_tier2.py           # Ablation + radar overview charts
├── plot_tier1.py           # Best-vs-best + best-overall charts
├── EVALUATION_GUIDE.md     # Tài liệu này
├── data_wandb/
│   └── all_runs_summary.csv
└── charts/                 # Auto-saved figures (Figure 01. Title.png, ...)
```

## Thứ tự chạy và câu chuyện phân tích

```
1. Load data                     → Biết được 24 configs cần đánh giá
          ↓
2. Histogram & Heatmap overview  → Thấy bức tranh toàn cảnh 24 configs
          ↓
3. Encoder Breakdown             → CLIP vs MBNv2: encoder nào mạnh hơn?
          ↓
4. Tầng 2 Ablation               → Từng model: sim_type nào tốt nhất?
          ↓
5. Tier 1 Preview                → Best config mỗi model trông như thế nào?
          ↓
6. Radar Overview (24 / 12 / 12) → Toàn cảnh "hình dạng" performance trước khi thu hẹp
          ↓
7. Tầng 1 Best-vs-Best @ K=5     → Model nào chiến thắng? (tiêu chí sát thực)
          ↓
8. Tầng 1 Best-vs-Best @ K=10    → Model nào chiến thắng? (so sánh với literature)
          ↓
9. Best Overall per Metric       → Kết luận tổng thể: model nào tốt nhất mỗi metric? ← Kết luận cuối
          ↓
10. Custom charts                → Drill-down bất kỳ thứ gì cần làm rõ thêm
```
