# Mã Giả Tinh Gọn (Pseudo-code v2) của 3 Mô hình

Tài liệu này tóm tắt luồng tính toán cốt lõi của 3 mô hình **CombiGCN**, **FREEDOM**, và **BM3** một cách ngắn gọn nhất, loại bỏ các chi tiết kỹ thuật phức tạp (như khởi tạo trọng số, L2 regularization, dropout) để bạn dễ hiểu và so sánh cấu trúc.

*Số chiều biểu diễn thống nhất: **512 chiều**.*

---

## 1. [CombiGCN](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/lightgcn_pyg/models/combigcn.py#L18) (Dual-Graph Propagation)
> **Ý tưởng:** Lan truyền đồng thời trên đồ thị tương tác $G_{\text{interaction}}$ và đồ thị tương đồng sản phẩm $G_{\text{similarity}}$ (đã tính offline).

```python
# ── [MA TRẬN TƯƠNG ĐỒNG W (S) LẤY TỪ ĐÂU?] ──
# Ma trận tương đồng W (trong công thức là S) KHÔNG được tính trong model mà được chuẩn bị trước bên ngoài:
# 1. Tính toán Offline: Trích xuất đặc trưng Raw (Ảnh/Chữ) -> Tính Cosine Similarity giữa các Items.
# 2. Chuẩn hóa đối xứng tạo thành ma trận kề thưa S.
# 3. Nạp vào qua train.py dưới dạng tham số `similarity_adj` khi gọi hàm forward().

Mô_hình CombiGCN:
    # ── [HÀM KHỞI TẠO (INIT)] ──
    def __init__(n_users, n_items, embedding_dim=512):
        # Khởi tạo ma trận nhúng ID cho User và Item [512 chiều]
        self.user_embedding = Embedding(n_users, 512)   # Học bằng Gradient
        self.item_embedding = Embedding(n_items, 512)   # Học bằng Gradient

    # ── [VÒNG LẶP HUẤN LUYỆN (FORWARD)] ──
    # Nhận vào: interaction_adj (đồ thị tương tác) và similarity_adj (Ma trận W/S ở trên)
    def forward(interaction_adj, similarity_adj, users, pos_items, neg_items):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # 1. Lan truyền đồ thị (Propagation) qua L layers
        ego = Concat([user_emb, item_emb], dim=0)
        all_layers = [ego]
        
        Cho layer từ 1 đến L:
            # Nhánh 1: User-Item CF Branch — Graph Conv (User-Item)
            interaction_emb = interaction_adj @ ego
            user_next = interaction_emb[Users]
            item_inter = interaction_emb[Items]       # Đầu ra từ Graph Conv (User-Item)
            
            # Nhánh 2: Item-Item Sim Branch — Matrix Mult (S @ Item Emb)
            # S chính là ma trận similarity_adj truyền vào từ bên ngoài
            item_sim = similarity_adj @ ego[Items]    # Phép nhân ma trận (Matrix Mult)
            
            # Dung hợp: Item Fusion (Sum)
            item_next = item_inter + item_sim          # Cộng hợp hai luồng thông tin
            
            # Ghép lại chuẩn bị cho layer tiếp theo
            ego = Concat([user_next, item_next], dim=0)
            all_layers.append(ego)
            
        # 2. Đầu ra cuối cùng (Mean Pooling qua các tầng)
        final_user, final_item = Mean(all_layers)
        
        # 3. Tính BPR Loss
        loss = BPR_Loss(final_user, final_item)
        return loss
```

---

## 2. [FREEDOM](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/lightgcn_pyg/models/freedom.py#L99) (Denoising & kNN Graph Denoising)
> **Ý tưởng:** Tách biệt hoàn toàn nhánh tương tác (CF Branch) và nhánh nội dung (Content Branch). Dùng **InfoNCE Loss** để khớp biểu diễn giữa 2 nhánh.

```python
Mô_hình FREEDOM:
    # ── [HÀM KHỞI TẠO (INIT)] ──
    def __init__(n_users, n_items, image_feats, text_feats):
        # 1. Khởi tạo ID embeddings
        self.user_embedding = Embedding(n_users, 512)
        self.item_embedding = Embedding(n_items, 512)
        
        # 2. Bộ chiếu tuyến tính đặc trưng thô
        self.image_projector = Linear(D_img, 512)
        self.text_projector = Linear(D_txt, 512)
        
        # 3. Tự dựng đồ thị kNN Item-Item từ Raw features lúc khởi tạo và ĐÓNG BĂNG
        self.knn_graph = Build_kNN_Graph_Offline(image_feats, text_feats, k=10) # Frozen

    # ── [VÒNG LẶP HUẤN LUYỆN (FORWARD)] ──
    def forward(interaction_adj, users, pos_items, neg_items):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Nhánh 1: User-Item CF Branch — GCN tương tác hành vi
        user_cf, item_cf = LightGCN_Propagate(interaction_adj, user_emb, item_emb)
        
        # Nhánh 2: Content Branch — GCN trên đồ thị kNN nội dung đóng băng
        # Chiếu ảnh/chữ về 512 chiều
        modal_emb = (self.image_projector(image_feats) + self.text_projector(text_feats)) / 2.0
        # Lan truyền trên đồ thị kNN đóng băng
        item_content = GCN_Propagate(self.knn_graph, modal_emb)
        
        # Đầu ra cuối cùng: Item Fusion (Cộng đặc trưng)
        final_item = item_cf + item_content
        
        # Tính Loss
        bpr_loss = BPR_Loss(user_cf, final_item)
        # InfoNCE: Đẩy biểu diễn CF và Content của sản phẩm tích cực lại gần nhau
        cl_loss = InfoNCE_Loss(item_cf, item_content) 
        
        loss = bpr_loss + cl_weight * cl_loss
        return loss
```

---

## 3. [BM3](file:///e:/DoCode/CD2/source/Source/get_hrs_rs/rs/lightgcn_pyg/models/bm3.py#L33) (Bootstrap Latent Representations)
> **Ý tưởng:** Không chạy GCN trên đặc trưng nội dung để tăng tốc. Dùng **target encoder cập nhật chậm (EMA)** làm mỏ neo, ép nhánh online dự đoán qua mạng **Predictor** phi tuyến (bất đối xứng 1 chiều) để chống sụp đổ biểu diễn.

```python
Mô_hình BM3:
    # ── [HÀM KHỞI TẠO (INIT)] ──
    def __init__(n_users, n_items, image_feats, text_feats):
        # 1. Online Encoder: Học bằng Gradient
        self.user_embedding = Embedding(n_users, 512)
        self.item_embedding = Embedding(n_items, 512)
        
        # 2. Target Encoder: Bản sao động lượng của Online Item, ĐÓNG BĂNG gradient
        self.item_embedding_target = Copy(self.item_embedding)
        self.item_embedding_target.requires_grad = False
        
        # 3. Bộ chiếu tuyến tính đặc trưng thô
        self.image_projector = Linear(D_img, 512)
        self.text_projector = Linear(D_txt, 512)
        
        # 4. Mạng Predictor phi tuyến để học bất đối xứng 1 chiều (Học bằng Gradient)
        self.predictor = MLP_2_Layers(512 -> 512)

    # ── [VÒNG LẶP HUẤN LUYỆN (FORWARD)] ──
    def forward(interaction_adj, users, pos_items, neg_items):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Nhánh 1: Online CF (Có gradient)
        user_cf, item_cf = LightGCN_Propagate(interaction_adj, user_emb, item_emb)
        
        # Nhánh 2: Modal Branch (Chỉ chiếu tuyến tính trực tiếp, KHÔNG truyền đồ thị)
        item_modal = (self.image_projector(image_feats) + self.text_projector(text_feats)) / 2.0
        
        # Nhánh 3: Target CF (EMA, đóng băng gradient)
        _, item_target = LightGCN_Propagate(interaction_adj, user_emb, self.item_embedding_target.weight)
        
        # Đầu ra cuối cùng: Item Fusion (Cộng đặc trưng)
        final_item = item_cf + item_modal
        
        # Tính Loss
        bpr_loss = BPR_Loss(user_cf, final_item)
        
        # Bootstrap Contrastive Loss (Bất đối xứng qua Predictor, target luôn Detach)
        # - CF dự đoán Modal
        loss_a = Cosine_Distance(self.predictor(item_cf), item_modal.detach())
        # - Modal dự đoán Target CF
        loss_b = Cosine_Distance(self.predictor(item_modal), item_target.detach())
        cl_loss = (loss_a + loss_b) / 2.0
        
        # Cập nhật động lượng EMA cho Target Encoder (đầu vào cho bước train sau)
        self.item_embedding_target.weight = 0.995 * self.item_embedding_target.weight + 0.005 * item_emb
        
        loss = bpr_loss + cl_weight * cl_loss
        return loss
```
