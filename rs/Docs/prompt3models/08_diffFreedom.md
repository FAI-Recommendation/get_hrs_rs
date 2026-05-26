## "Lý do thiết kế" cho từng điểm khác paper 😏

---

### 1. Concat features → build 1 graph (thay vì build riêng rồi trộn α)

**Nói thế này:**

> Paper dùng hyperparameter α_v cố định để trộn 2 graph (S = α_v·S^v + α_t·S^t). Cách này có 2 vấn đề: **(a)** phải tune thêm α — với dataset nhỏ (2194 items) dễ overfit vào 1 giá trị α cụ thể, **(b)** build 2 graph riêng tốn O(2·N²·d) rồi cộng lại, trong khi concat rồi build 1 lần chỉ tốn O(N²·(d_v+d_t)) — cùng complexity nhưng bỏ được 1 hyperparameter. Ngoài ra, concat features trước khi tính cosine similarity cho phép **cross-modal interaction** ngay tại bước similarity — 2 items có thể giống nhau ở tổng thể (image + text cùng lúc) dù mỗi modal riêng lẻ không đủ giống.

---

### 2. Weighted graph + row-normalize (thay vì unweighted 0/1 + symmetric norm)

**Nói thế này:**

> Paper convert sang unweighted (0 hoặc 1), nghĩa là **mất hết thông tin mức độ giống nhau** — item giống 0.95 và item giống 0.51 đều được coi ngang nhau. Giữ weighted graph **bảo toàn similarity strength**, neighbor giống nhiều sẽ đóng góp nhiều hơn khi propagate — phù hợp hơn với bài toán fashion vì mức độ giống nhau giữa sản phẩm rất quan trọng (áo đỏ giống áo cam hơn áo xanh).
>
> Row-normalize thay vì symmetric normalize vì row-normalize đảm bảo **mỗi hàng tổng = 1** — mỗi item nhận trung bình có trọng số từ neighbors, tránh scale khác nhau giữa items có degree khác nhau. Symmetric normalize phù hợp hơn cho bipartite graph (user-item) nơi 2 bên có degree rất khác nhau, nhưng kNN graph mỗi node đều có đúng k=10 neighbors nên row-normalize đơn giản và ổn định hơn.

---

### 3. Random dropout thay vì degree-sensitive denoising

**Nói thế này:**

> Degree-sensitive pruning trong paper thiết kế cho dataset lớn (Amazon, Yelp — hàng trăm nghìn users) nơi **degree distribution rất skewed** (power-law) — popular items có hàng nghìn interactions, dễ bị over-smoothing. Dataset của mình chỉ có **553 users, 2194 items, ~9455 interactions** — trung bình mỗi user chỉ ~17 interactions, degree distribution tương đối đều, không có hiện tượng "super popular node" nghiêm trọng. Trong trường hợp này, degree-sensitive pruning gần như tương đương random dropout. Mình chọn random dropout để **giảm complexity implementation** mà không hy sinh hiệu quả trên dataset nhỏ.

---

### 4. Modal features làm input content propagation (thay vì ID embeddings)

**Đây là điểm HAY NHẤT để defend:**

> Paper dùng ID embeddings h⁰ᵢ làm input cho item-item graph propagation. Nhưng kNN graph được build từ **modal features** (image/text similarity) — tức graph structure encode thông tin content. Nếu propagate ID embeddings qua graph này thì ý nghĩa là: *"items giống nhau về visual/text nên có ID embeddings giống nhau"* — đây là **indirect alignment**.
>
> Implementation của mình propagate **modal features trực tiếp** qua kNN graph: *"items giống nhau về visual/text nên chia sẻ và làm giàu thêm thông tin content cho nhau"* — đây là **direct content enrichment**. Cách này **semantically consistent** hơn: graph build từ content, thì nên propagate content qua nó. Tương tự cách GraphSAGE aggregate features từ neighbors — mỗi item "hấp thụ" đặc trưng visual/text từ items giống nó, tạo ra biểu diễn content phong phú hơn trước khi fuse với CF view.

---

### 5. Mean pooling all layers (thay vì chỉ lấy layer cuối)

**Nói thế này:**

> Paper lấy layer cuối h^(L_ii), nghĩa là chỉ giữ thông tin sau khi đã propagate xa nhất. Điều này có rủi ro **over-smoothing** — qua 4 layers, tất cả items có thể converge về cùng 1 representation. Mean pooling theo convention của LightGCN giữ lại **multi-scale information**: layer 0 = bản thân item, layer 1 = neighbor trực tiếp, layer 2 = neighbor của neighbor, v.v. Mỗi scale mang thông tin khác nhau. Cách này đã được chứng minh hiệu quả trong LightGCN paper và nhiều GCN-based recommendation systems.

---

### 6. InfoNCE thay vì modal-specific BPR

**Nói thế này:**

> Paper dùng BPR loss riêng cho từng modality (Eq 10): xếp hạng user-item dựa trên visual features riêng, text features riêng. Cách này có hạn chế: **(a)** mỗi modal BPR hoạt động **độc lập**, không có cơ chế explicit alignment giữa CF view và content view, **(b)** ranking loss chỉ push positive > negative, không tạo ra **representation alignment** trực tiếp.
>
> InfoNCE contrastive loss giữa CF view và content view **explicitly aligns** 2 biểu diễn của cùng 1 item: item_cf[i] phải giống item_content[i] (positive) và khác item_content[j≠i] (negative). Đây là **cross-view alignment** mạnh hơn — đảm bảo 2 nhánh (CF + Content) học được biểu diễn **complementary** chứ không contradictory. Cách này cũng consistent với xu hướng contrastive learning trong recommendation (SGL, SimGCL, NCL) cho kết quả tốt hơn modal-specific ranking.

---

### 7. Prediction dùng cả content (thay vì chỉ ID)

**Nói thế này:**

> Paper note rằng thêm modal features vào prediction không cải thiện trên dataset của họ. Tuy nhiên, dataset fashion của mình có đặc thù: **553 users chỉ có ~17 interactions mỗi người** — rất sparse. Trong trường hợp này, ID embeddings chưa được train đủ để capture hết preference. Content embeddings (đã propagate qua kNN graph) mang thêm thông tin visual/text giúp **bù đắp sparsity** — đặc biệt quan trọng cho items có ít interactions (near cold-start). Đây là trade-off giữa paper (dataset lớn, ID đủ mạnh) vs implementation (dataset nhỏ, cần thêm content signal).

---

## Tóm lại — "câu chuyện" tổng thể nếu bị hỏi:

> *"Implementation lấy cảm hứng từ FREEDOM paper nhưng được **adapted cho đặc thù dataset fashion nhỏ** (553 users). Các thay đổi chính: (1) đơn giản hóa graph construction bằng cách bỏ hyperparameter α, (2) giữ weighted graph để bảo toàn thông tin similarity, (3) bỏ degree-sensitive pruning vì dataset nhỏ không có skewed degree, (4) propagate modal features thay vì ID embeddings cho semantic consistency, (5) dùng InfoNCE cho explicit cross-view alignment mạnh hơn modal BPR, và (6) mean pooling theo LightGCN convention để tránh over-smoothing. Mỗi thay đổi đều có lý do kỹ thuật cụ thể, không phải bỏ sót."*

Bạn muốn mình ghi những lý do này vào docs luôn không? 😄