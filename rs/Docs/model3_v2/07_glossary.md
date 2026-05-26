# 07 — Glossary (Tu dien thuat ngu)

## A. Kien truc & Mo hinh

| Thuat ngu | Giai thich |
|---|---|
| **LightGCN** | Mo hinh GCN don gian nhat cho recommendation: chi dung phep nhan ma tran (adj @ emb) de truyen thong tin giua user-item, khong co activation function hay weight matrix. La "backbone" cua ca 3 models trong project nay. |
| **GCN** (Graph Convolutional Network) | Mang neural hoat dong tren do thi. Moi node "hap thu" thong tin tu cac neighbors cua no qua cac layers. |
| **CombiGCN** | Mo hinh ket hop 2 do thi: user-item interaction graph + item-item similarity graph. "Combi" = Combination. |
| **BM3** (Bootstrap Multi-modal Model) | Mo hinh dung bootstrap contrastive learning (kieu BYOL) de hoc alignment giua CF embeddings va multimodal embeddings. Khong can negative samples. Paper: WWW 2023. |
| **FREEDOM** (Freezing and Denoising) | Mo hinh dong bang (freeze) item-item kNN graph roi truyen thong tin content tren graph do. Dung InfoNCE contrastive loss. Paper: ACM MM 2023. |
| **BYOL** (Bootstrap Your Own Latent) | Phuong phap self-supervised learning cua DeepMind. Hoc bieu dien ma khong can negative samples — chi can online network + target network (EMA). BM3 lay y tuong tu day. |

## B. Do thi & Ma tran

| Thuat ngu | Giai thich |
|---|---|
| **Bipartite graph** | Do thi 2 phia: 1 ben la users, 1 ben la items. Canh noi user-item neu user da mua/tuong tac voi item do. |
| **interaction_adj** | Ma tran ke (adjacency matrix) cua bipartite graph. Kich thuoc [n_users+n_items, n_users+n_items]. Da duoc normalize. |
| **similarity_adj** | Ma tran tuong dong item-item. Kich thuoc [n_items, n_items]. Chi CombiGCN dung. |
| **kNN graph** | Do thi k-nearest neighbors: moi item noi voi k items giong no nhat (do bang cosine similarity). FREEDOM dung, build 1 lan roi freeze. |
| **Adjacency matrix** (ma tran ke) | Ma tran vuong the hien cau truc do thi: A[i][j] = 1 neu co canh tu node i den node j, = 0 neu khong. |
| **Sparse matrix** | Ma tran thua — phan lon gia tri = 0. Luu tru hieu qua bang chi giu cac vi tri != 0. Dung scipy.sparse hoac torch_sparse. |
| **SparseTensor** | Kieu du lieu cua thu vien torch_sparse. Hieu qua hon PyTorch native sparse. Dung matmul() de nhan. |
| **D^(-0.5) A D^(-0.5)** | Symmetric normalization: chia moi canh cho can bac 2 cua degree 2 dau. Giup on dinh gradient khi truyen nhieu layers. D = degree matrix. |
| **Cosine similarity** | Do tuong dong giua 2 vector: cos(a,b) = (a·b) / (‖a‖·‖b‖). Gia tri tu -1 (nguoc) den 1 (giong). |
| **Threshold 0.5** | Loai bo nhung cap item co similarity < 0.5 (coi nhu khong lien quan). Giam noise trong similarity matrix. |

## C. Embeddings & Features

| Thuat ngu | Giai thich |
|---|---|
| **Embedding** | Vector so bieu dien 1 doi tuong (user hoac item) trong khong gian nhieu chieu. VD: user_emb [512] la 1 vector 512 chieu dai dien cho 1 user. |
| **ID embedding** | Embedding hoc tu du lieu tuong tac (ai mua gi). Moi user/item co 1 vector rieng, khoi tao ngau nhien, hoc qua training. |
| **Modal embedding** | Embedding tu dac trung noi dung (image/text) cua item. Khac ID embedding — no mang thong tin "item nay trong nhu the nao, mo ta ra sao". |
| **ego_emb** | Ket noi (concat) user_emb va item_emb thanh 1 tensor lon [n_users+n_items, dim]. "Ego" = ban than node truoc khi truyen tin. |
| **Xavier init** | Cach khoi tao trong so: random tu phan phoi deu, scale theo so input/output. Giup gradient on dinh tu dau. Ten tu Glorot Xavier. |
| **Projector** (Linear projection) | 1 layer Linear(dim_in → dim_out) de chuyen doi embedding tu khong gian nay sang khong gian khac. VD: image 512 chieu → embedding 512 chieu. |
| **register_buffer** | Cach PyTorch luu tensor vao model ma KHONG tinh gradient. Tensor se di theo model.to(device) nhung khong duoc update boi optimizer. Dung cho: frozen features, frozen graph. |

## D. Dac trung da phuong thuc (Multimodal)

| Thuat ngu | Giai thich |
|---|---|
| **Multimodal** | Su dung nhieu nguon thong tin (modality) cung luc. O day: image (hinh anh) + text (mo ta san pham). |
| **sim_type** | Tham so chon cach ket hop features: `img_only` (chi anh), `tfidf` (chi text), `multimodal` (ca 2, trung binh), `multimodal_attention` (ca 2, hoc weight). |
| **Late fusion** | Ket hop 2 modal SAU khi da xu ly rieng: tinh img_emb va txt_emb rieng, roi trung binh: (img + txt) / 2. Don gian, hieu qua. |
| **Attention fusion** | Ket hop 2 modal bang cach HOC weight: concat(img, txt) → Linear → output. Model tu quyet dinh img hay txt quan trong hon. |
| **CLIP** | Mo hinh cua OpenAI, hoc joint embedding cua image + text. Dung de trich dac trung anh. Output: vector 512 chieu. |
| **MobileNetV2** (MBNv2) | Mo hinh CNN nhe cua Google, thiet ke cho mobile. Dung de trich dac trung anh. Nap bat texture/hoa tiet tot hon CLIP cho thoi trang. |
| **TF-IDF** | (Term Frequency - Inverse Document Frequency) Phuong phap bieu dien text bang vector so. Tu xuat hien nhieu trong 1 document nhung it trong corpus → trong so cao. |
| **BERT** | Mo hinh ngon ngu cua Google. Hieu ngu canh 2 chieu (bidirectional). Dung de trich dac trung text. Output: vector 768 chieu. |
| **Encoder** (trong boi canh nay) | Mo hinh dung de trich dac trung anh: CLIP hoac MobileNetV2. Khong phai encoder trong Transformer. |

## E. Loss Functions

| Thuat ngu | Giai thich |
|---|---|
| **BPR loss** (Bayesian Personalized Ranking) | Loss cho recommendation: ep diem cua item user THICH phai cao hon item user KHONG THICH. Cong thuc: softplus(-(pos_score - neg_score)). |
| **softplus** | Ham: softplus(x) = log(1 + e^x). Phien ban "muot" cua ReLU. Dung trong BPR de tranh gradient = 0. |
| **L2 regularization** | Phat mo hinh neu weights qua lon: reg = decay × ‖w‖². Ngan overfit. `decay` = he so phat (1e-4). |
| **Contrastive loss** | Loss ep 2 bieu dien cua CUNG 1 doi tuong (positive pair) gan nhau, va KHAC doi tuong (negative pair) xa nhau. |
| **Bootstrap loss** | Loai contrastive KHONG CAN negative samples. Chi ep online prediction gan target (EMA). Cong thuc: 2 - 2·cos(pred, target). Dung trong BM3. |
| **InfoNCE** | Loai contrastive CAN negative samples. Coi diagonal cua similarity matrix la positives, off-diagonal la negatives. Dung CrossEntropy. Dung trong FREEDOM. |
| **Temperature** (tau, τ) | He so chia trong InfoNCE: logits = sim / tau. Tau nho → phan bo "nhon" (phan biet manh), tau lon → phan bo "det" (de hoc). Mac dinh: 0.2. |

## F. Training

| Thuat ngu | Giai thich |
|---|---|
| **BPR sampling** | Moi step: chon 1 batch users, moi user lay 1 positive item (da mua) va 1 negative item (chua mua). |
| **Positive item** | Item ma user DA tuong tac (mua, click, ...). |
| **Negative item** | Item ma user CHUA tuong tac. Gia dinh la user khong thich (co the sai, nhung du tot cho training). |
| **EMA** (Exponential Moving Average) | Cap nhat tham so cham: target = 0.995 × target + 0.005 × online. Tao ra phien ban "on dinh" cua model. Dung trong BM3. |
| **Momentum** | He so EMA (0.995). Gan 1 = cap nhat rat cham (target on dinh). Gan 0 = cap nhat nhanh (target giong online). |
| **Predictor head** | MLP nho (Linear → ReLU → Linear) chi dung trong bootstrap loss. Tao "asymmetry" giua online va target — ngan collapse. |
| **Collapse** | Hien tuong tat ca embeddings hoi tu ve cung 1 diem (model "luoi" — tra ve giong nhau cho moi input). EMA + predictor ngan dieu nay. |
| **AMP** (Automatic Mixed Precision) | Dung fp16 (16-bit) thay vi fp32 de tinh toan nhanh hon ~2x, it VRAM hon. PyTorch tu dong chon cho nao dung fp16 an toan. |
| **GradScaler** | Di kem AMP: scale gradient len truoc khi backward (tranh underflow o fp16), roi scale xuong truoc khi update weights. |
| **Early stopping** | Dung training som neu metric khong cai thien sau N lan eval lien tiep. Ngan overfit. |
| **Epoch** | 1 lan duyet qua toan bo training data. 1000 epochs = duyet 1000 lan. |
| **Eval interval** | Danh gia model moi N epochs (N=40). Khong eval moi epoch vi ton thoi gian. |

## G. Evaluation Metrics

| Thuat ngu | Giai thich |
|---|---|
| **Recall@K** | Trong K items duoc recommend, bao nhieu % ground truth (items user thuc su thich) duoc tim thay? Recall@10 = 0.05 nghia la tim duoc 5% items dung trong top 10. |
| **Precision@K** | Trong K items recommend, bao nhieu % la dung? Precision@10 = 0.01 nghia la 1 trong 10 items la dung. |
| **NDCG@K** (Normalized Discounted Cumulative Gain) | Do chat luong ranking: item dung o vi tri cao duoc thuong nhieu hon vi tri thap. Gia tri 0-1, 1 = perfect ranking. |
| **MAP@K** (Mean Average Precision) | Trung binh precision tai moi vi tri co hit. Phat model neu item dung bi day xuong vi tri thap. |
| **MRR@K** (Mean Reciprocal Rank) | 1 / vi_tri_item_dung_dau_tien. MRR = 0.5 nghia la trung binh item dung nam o vi tri thu 2. |
| **Hit Ratio@K** | Co IT NHAT 1 item dung trong top-K khong? 1 = co, 0 = khong. Metric "de tinh" nhat. |
| **K** | So items recommend (top-K). K=1 (chi 1 item), K=5, K=10, K=20. |
| **Ground truth** | Tap items ma user THUC SU tuong tac trong test set. Day la "dap an dung" de so sanh voi recommendations. |

## H. Propagation & Graph Operations

| Thuat ngu | Giai thich |
|---|---|
| **Propagation** | Truyen thong tin qua do thi: moi node cap nhat embedding = tong (co trong so) embedding cua neighbors. Qua nhieu layers → nhan thong tin tu xa hon. |
| **Layer** | 1 lan truyen tin. Layer 1: neighbor truc tiep. Layer 2: neighbor cua neighbor. 4 layers = thong tin truyen xa 4 buoc. |
| **matmul(adj, emb)** | Phep nhan sparse matrix × dense matrix. Day la toan bo "graph convolution" trong LightGCN — khong co gi phuc tap hon. |
| **Mean pooling** | Lay trung binh tat ca layers: final = (layer0 + layer1 + ... + layerK) / (K+1). Giu thong tin tu moi muc do "xa". |
| **Node dropout** | Random bo 1 so canh trong do thi khi training (de regularize). Dropout rate 0.1 = bo 10% canh. Mac dinh: tat. |
| **Frozen** (dong bang) | Khong cho gradient chay qua → khong update khi training. FREEDOM dong bang kNN graph. BM3 dong bang target encoder (chi update qua EMA). |
| **Content propagation** | (FREEDOM rieng) Truyen modal embeddings qua frozen kNN graph. Items "hap thu" dac trung tu neighbors giong no. |
| **CF propagation** | (Collaborative Filtering) Truyen embeddings qua user-item interaction graph. Hoc tu hanh vi tuong tac. |

## I. Cong cu & Infra

| Thuat ngu | Giai thich |
|---|---|
| **WandB** (Weights & Biases) | Platform log experiment online. Theo doi loss, metrics, hyperparams qua web dashboard. |
| **TensorBoard** | Cong cu cua TensorFlow (dung duoc voi PyTorch) de visualize loss/metrics. Chay local. |
| **HuggingFace Hub** | Platform luu tru models online. Push best_model.pt len de chia se hoac deploy. |
| **torch_sparse** | Thu vien toi uu cho sparse matrix operations tren GPU. Nhanh hon PyTorch native sparse. |
| **scipy.sparse** | Thu vien Python cho sparse matrix (CPU). Dung de build va cache adjacency matrices (.npz files). |
| **.npz** | Format nen cua numpy/scipy de luu sparse matrix. Dung de cache similarity matrices — build 1 lan, load nhanh lan sau. |
| **.npy** | Format numpy de luu dense array. Dung cho image_embeddings.npy va text_embeddings.npy. |
| **AMP fp16** | Automatic Mixed Precision: dung float16 thay float32 cho 1 so phep tinh. Nhanh ~2x, it VRAM. |
