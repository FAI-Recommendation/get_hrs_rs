# Evaluation Guide — So sánh Models cho Chuyên đề 2

Notebook `evaluate.ipynb` đánh giá và so sánh hiệu năng của các mô hình gợi ý (recommendation) dựa trên dữ liệu từ WandB. Phân tích đi từ **tổng quan** → **phân tách theo encoder** → **ablation từng model** → **so sánh trực tiếp best-vs-best**.

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
   Dùng NDCG@10 làm tiêu chí chính.

3. **Ranking table** (`display_ranking_table`):  
   DataFrame styled với highlight ô max, sort theo NDCG@10 giảm dần.

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
> → Nếu thay đổi: model tốt ở top-1 chưa chắc tốt ở top-20 (và ngược lại). Cần xác định use case: precision (K nhỏ) hay recall (K lớn). → Sang Tầng 1 Best-vs-Best để so sánh chi tiết qua K.

---

## Tầng 1 — Best-vs-Best: Model nào tốt nhất?

```python
best_df, best_results = plot_best_vs_best(df)
display(display_summary_table(best_df))
```

Lấy config tốt nhất của mỗi model (theo NDCG@10) rồi so sánh trực tiếp.

**Best configs được chọn:**

| Model | Best config | NDCG@10 |
|---|---|---|
| bm3 | `mbnv2(multimodal)` | 0.018595 |
| combigcn | `mbnv2(multimodal)` | 0.017486 |
| freedom | `mbnv2(multimodal_attention)` | 0.008759 |

**3 loại chart:**

1. **Lineplot grid** (`plot_lineplot_grid`):  
   Grid **2×3**, 6 metrics. Mỗi subplot: X = K, Y = metric, mỗi đường = một model.  
   Đường thắng highlight, ★ đỏ + annotation tên tại điểm tốt nhất mỗi K.

2. **Single lineplot** (`plot_lineplot_single`):  
   Phóng to chart NDCG@K riêng lẻ — rõ hơn khi cần trình bày.

3. **Radar chart** (`plot_radar_chart`):  
   So sánh tổng thể trên 6 metrics (trung bình qua K). Diện tích radar = sức mạnh tổng hợp.

**Summary table:**  
Styled DataFrame với highlight ô max per column. Rank 1 = model tốt nhất.

> **Q&A — Sau Best-vs-Best**
>
> **Q: Khoảng cách giữa BM3 (rank 1) và CombiGCN (rank 2) có đáng kể không?**  
> → Nếu < 5% relative difference: 2 model gần tương đương, lựa chọn có thể dựa trên tốc độ inference hoặc tính ổn định training. Nếu > 10%: BM3 rõ ràng vượt trội.
>
> **Q: FREEDOM kém BM3 bao nhiêu? Có metric nào FREEDOM không quá tệ không?**  
> → Nếu FREEDOM gần hơn trên Precision@1: nó có thể phù hợp hơn cho use case cần top-1 chính xác. Radar chart sẽ cho thấy "hình dạng" strength/weakness của mỗi model.
>
> **Q: Thứ hạng model có nhất quán qua K=1, 5, 10, 20 không?**  
> → Nếu model A thắng K=1 nhưng model B thắng K=20: cần chọn K phù hợp với product (gợi ý 1 item vs top-20). → Kết quả này là câu trả lời cuối cùng cho câu hỏi "Model nào tốt nhất cho Chuyên đề 2?"

---

## Custom: Vẽ riêng từng chart

Cell cuối notebook chứa ví dụ để vẽ/save từng loại chart riêng lẻ khi cần customize chi tiết:

```python
# Heatmap một model + metric cụ thể
plot_heatmap_per_model(df, model="combigcn", metric="recall")

# Single lineplot NDCG phóng to
plot_lineplot_single(best_results, metric="ndcg")

# Save toàn bộ ra folder
plot_ablation(df, save_dir="charts/tier2")
plot_best_vs_best(df, save_dir="charts/tier1")
```

> **Q&A — Hướng phân tích tiếp theo**
>
> **Q: Nếu thêm model mới (VD: SGL, SimGCL), cần làm gì?**  
> → Thêm runs vào CSV, gọi lại `load_from_csv()`. Tất cả hàm plot sẽ tự động include model mới — không cần sửa code plot.
>
> **Q: Nếu muốn so sánh theo metric khác thay vì NDCG@10 làm rank?**  
> → Truyền `rank_metric="recall@10"` vào `plot_best_vs_best(df, rank_metric="recall@10")`. Best config được chọn sẽ thay đổi theo metric này.
>
> **Q: Muốn export bảng kết quả ra LaTeX/CSV cho báo cáo?**  
> → `summary_table(best_df).to_latex()` hoặc `.to_csv("results.csv")`. Hàm `summary_table` trả về DataFrame thuần, không phụ thuộc matplotlib.

---

## Cấu trúc file

```
data_evaluate/
├── evaluate.ipynb          # Notebook chính (19 cells)
├── load_data.py            # Load & transform data từ WandB CSV
├── plot_tier2.py           # Ablation charts (heatmap, barplot, histogram, lineplot)
├── plot_tier1.py           # Best-vs-best charts (lineplot grid, radar, histogram)
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
6. Tầng 1 Best-vs-Best           → Model nào chiến thắng?  ← Kết luận chính
          ↓
7. Custom charts                 → Drill-down bất kỳ thứ gì cần làm rõ thêm
```
