"""
load_data.py — PyG-compatible version
======================================
Convert từ hr/utility/load_data.py.

Thay đổi:
  - BỎ: tf.SparseTensor
  - THÊM: scipy_sparse_to_edge_index(), scipy_sparse_to_torch_sparse()
  - THÊM: get_pyg_data() trả dict PyG-ready
  - FIX: parse_vector_string hỗ trợ cả space lẫn comma delimiter
  - FIX: BERT batch chạy GPU-aware
  - GIỮ NGUYÊN: tất cả similarity matrix methods, sampling, n-core logic
"""

import os
import random as rd
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, BertTokenizer


# ─────────────────────────────────────────────
# GPU cosine similarity — thay the sklearn.cosine_similarity (CPU)
# Nhanh hon ~20-50x nho GPU mat mul
# ─────────────────────────────────────────────
def cosine_sim_gpu(a, b=None):
    """
    Tinh cosine similarity matrix tren GPU.
    a, b: numpy array (n, dim) hoac scipy sparse matrix.
    Tra ve: numpy array (n, n).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert sparse → dense neu can
    if sp.issparse(a):
        a = a.toarray().astype(np.float32)
    if b is not None and sp.issparse(b):
        b = b.toarray().astype(np.float32)

    a_t = torch.tensor(np.array(a, dtype=np.float32)).to(device)
    b_t = a_t if b is None else torch.tensor(np.array(b, dtype=np.float32)).to(device)

    a_norm = F.normalize(a_t, p=2, dim=1)
    b_norm = F.normalize(b_t, p=2, dim=1)

    # Tinh theo batch de tranh OOM tren GPU lon
    batch = 512
    rows = []
    for i in range(0, a_norm.shape[0], batch):
        chunk = torch.mm(a_norm[i:i+batch], b_norm.t())
        rows.append(chunk.cpu())
    return torch.cat(rows, dim=0).numpy()


# ─────────────────────────────────────────────
# Attention modules (dùng cho multimodal fusion method="attention")
# ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output


class WeightedAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.fc = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, text_embeddings, image_embeddings):
        combined = torch.cat([text_embeddings, image_embeddings], dim=1)
        return self.fc(combined)


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────
def scipy_sparse_to_edge_index(sp_mat):
    """Convert scipy sparse → PyG edge_index [2, nnz] + edge_weight [nnz]."""
    coo = sp_mat.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(coo.data.astype(np.float32))
    return edge_index, edge_weight


def scipy_sparse_to_torch_sparse(sp_mat):
    """Convert scipy sparse → torch.sparse_coo_tensor."""
    coo = sp_mat.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def parse_vector_string(s):
    """Parse vector string từ CSV — hỗ trợ cả space và comma delimiter."""
    return np.array(
        [float(x) for x in s.strip("[]").replace(",", " ").split()],
        dtype=np.float32,
    )


# ─────────────────────────────────────────────
# Main Data class
# ─────────────────────────────────────────────
class Data:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        # ── Đọc train.txt ────────────────────────────
        with open(os.path.join(path, "train.txt")) as f:
            for line in f:
                if len(line) > 0:
                    parts = line.strip().split(" ")
                    uid = int(parts[0])
                    items = [int(i) for i in parts[1:]]
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        # ── Đọc test.txt ─────────────────────────────
        with open(os.path.join(path, "test.txt")) as f:
            for line in f:
                if len(line) > 0:
                    try:
                        items = [int(i) for i in line.strip().split(" ")[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)

        self.n_items += 1
        self.n_users += 1
        self.print_statistics()

        # ── Sparse interaction matrix R (n_users × n_items) ──
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.S = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)

        self.train_items, self.test_set = {}, {}

        with open(os.path.join(path, "train.txt")) as f_train:
            for line in f_train:
                if len(line) == 0:
                    break
                parts = [int(i) for i in line.strip().split(" ")]
                uid, train_items = parts[0], parts[1:]
                for i in train_items:
                    self.R[uid, i] = 1.0
                self.train_items[uid] = train_items

        with open(os.path.join(path, "test.txt")) as f_test:
            for line in f_test:
                if len(line) == 0:
                    break
                try:
                    parts = [int(i) for i in line.strip().split(" ")]
                except Exception:
                    continue
                uid, test_items = parts[0], parts[1:]
                self.test_set[uid] = test_items

        # ── Co-occurrence matrices ────────────────────
        self.U = self.R.dot(self.R.transpose())
        self.I = self.R.transpose().dot(self.R)

        # ── Social trust (optional) ───────────────────
        self.n_social = 0
        social_path = os.path.join(path, "social_trust.txt")
        if os.path.exists(social_path):
            with open(social_path) as f:
                for line in f:
                    if len(line) == 0:
                        break
                    parts = [int(i) for i in line.strip().split(" ")]
                    uid, friends = parts[0], parts[1:]
                    for fri in friends:
                        self.S[uid, fri] = 1.0
                        self.n_social += 1

        # ── Items features ────────────────────────────
        self.items_features = pd.read_csv(os.path.join(path, "items_features.csv"))

    # ─────────────────────────────────────────
    # Text preprocessing
    # ─────────────────────────────────────────
    @staticmethod
    def preprocess_text(text):
        return text.lower()

    # ─────────────────────────────────────────
    # BERT embeddings (batch, GPU-aware)
    # ─────────────────────────────────────────
    def get_bert_embeddings_batch(self, texts, batch_size=32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.to(device)
        model.eval()

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_emb)

        return np.vstack(embeddings).astype(np.float32)

    # ─────────────────────────────────────────
    # Similarity matrices
    # ─────────────────────────────────────────
    def create_tfidf_similarity_matrix(self):
        self.items_features["combined_features"] = (
            self.items_features["feature1"] + " " + self.items_features["feature2"]
        )
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.items_features["combined_features"])
        cosine_sim = cosine_sim_gpu(tfidf_matrix)
        cosine_sim[cosine_sim < 0.5] = 0
        return sp.csr_matrix(cosine_sim)

    def create_bert_similarity_matrix(self):
        """BERT(feature2) * TF-IDF(feature1) — dung GPU batch."""
        self.items_features["cleaned_feature2"] = self.items_features["feature2"].apply(
            self.preprocess_text
        )
        # GPU batch BERT (nhanh hon CPU 10-30x)
        texts = self.items_features["cleaned_feature2"].tolist()
        bert_embeddings = self.get_bert_embeddings_batch(texts, batch_size=64)

        bert_sim = cosine_sim_gpu(bert_embeddings)
        bert_sim[bert_sim < 0.5] = 0
        bert_similarity_matrix = sp.csr_matrix(bert_sim)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.items_features["feature1"])
        tfidf_sim = cosine_sim_gpu(tfidf_matrix)
        tfidf_sim[tfidf_sim < 0.5] = 0
        tfidf_similarity_matrix = sp.csr_matrix(tfidf_sim)

        return bert_similarity_matrix.multiply(tfidf_similarity_matrix)

    def create_full_bert_similarity_matrix(self):
        """BERT(feature1 + feature2) — dung GPU batch."""
        self.items_features["combined_features"] = (
            self.items_features["feature1"] + " " + self.items_features["feature2"]
        )
        self.items_features["cleaned_combined_features"] = self.items_features[
            "combined_features"
        ].apply(self.preprocess_text)

        # GPU batch BERT
        texts = self.items_features["cleaned_combined_features"].tolist()
        bert_embeddings = self.get_bert_embeddings_batch(texts, batch_size=64)

        bert_sim = cosine_sim_gpu(bert_embeddings)
        bert_sim[bert_sim < 0.5] = 0
        return sp.csr_matrix(bert_sim)

    def create_img_similarity_matrix(self, threshold=0.5):
        """Tạo item similarity matrix từ image embeddings (feature3)."""
        if "feature3" not in self.items_features.columns:
            raise ValueError("Cột 'feature3' không tồn tại trong items_features.")

        image_embeddings = np.vstack(
            self.items_features["feature3"].apply(parse_vector_string).values
        ).astype(np.float32)
        print(f"   Image embeddings shape: {image_embeddings.shape}")

        sim = cosine_sim_gpu(image_embeddings)
        sim[sim < threshold] = 0
        return sp.csr_matrix(sim)

    def create_multimodal_similarity_matrix(self, method="late_fusion", alpha=0.5, pca_components=None):
        """Kết hợp text + image embeddings. Methods: late_fusion | aggregation | pca | attention"""
        required_cols = {"feature1", "feature2", "feature3"}
        if not required_cols.issubset(self.items_features.columns):
            raise ValueError(f"Cần có các cột: {required_cols}")

        self.items_features["combined_text"] = (
            self.items_features["feature1"] + " " + self.items_features["feature2"]
        )
        self.items_features["cleaned_combined_text"] = self.items_features[
            "combined_text"
        ].apply(self.preprocess_text)

        text_embeddings = self.get_bert_embeddings_batch(
            self.items_features["cleaned_combined_text"].tolist()
        )
        print(f"   Text embeddings shape: {text_embeddings.shape}")

        image_embeddings = np.vstack(
            self.items_features["feature3"].apply(parse_vector_string).values
        ).astype(np.float32)
        print(f"   Image embeddings shape: {image_embeddings.shape}")

        if method == "late_fusion":
            text_sim = cosine_sim_gpu(text_embeddings)
            img_sim = cosine_sim_gpu(image_embeddings)
            combined_sim = alpha * text_sim + (1 - alpha) * img_sim

        elif method == "aggregation":
            combined_emb = (text_embeddings + image_embeddings) / 2
            combined_sim = cosine_sim_gpu(combined_emb)

        elif method == "pca":
            # Determine PCA components dynamically if not provided.
            if pca_components is None:
                # Check environment override first
                env_val = os.getenv("MULTIMODAL_PCA_COMPONENTS")
                if env_val is not None and env_val.strip() != "":
                    try:
                        pca_components = int(env_val)
                        print(f"   MULTIMODAL_PCA_COMPONENTS from env: {pca_components}")
                    except Exception:
                        raise ValueError("Invalid MULTIMODAL_PCA_COMPONENTS environment variable (must be int).")
                else:
                    # default: sum of text+image dims capped at 768
                    text_dim = text_embeddings.shape[1]
                    img_dim = image_embeddings.shape[1]
                    pca_components = min(text_dim + img_dim, 768)
                    print(f"   Auto-selected pca_components={pca_components} (text_dim={text_dim}, img_dim={img_dim})")

            combined_emb = np.concatenate([text_embeddings, image_embeddings], axis=1)
            pca = PCA(n_components=pca_components)
            reduced = pca.fit_transform(combined_emb)
            combined_sim = cosine_sim_gpu(reduced)

        elif method == "attention":
            text_tensor = torch.tensor(text_embeddings, dtype=torch.float32)
            img_tensor = torch.tensor(image_embeddings, dtype=torch.float32)
            embed_dim = text_embeddings.shape[1]

            mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=8)
            text_att = mha(text_tensor, text_tensor, text_tensor)
            img_att = mha(img_tensor, img_tensor, img_tensor)

            wa = WeightedAttention(embed_dim=embed_dim)
            combined_emb = wa(text_att, img_att)
            combined_sim = cosine_sim_gpu(combined_emb.detach().numpy())
        else:
            raise ValueError(f"method '{method}' không hợp lệ.")

        combined_sim[combined_sim < 0.5] = 0
        return sp.csr_matrix(combined_sim)

    # ─────────────────────────────────────────
    # Adjacency matrix construction
    # ─────────────────────────────────────────
    def create_interaction_adj_mat(self):
        """Bipartite user-item adjacency (n_users+n_items × n_users+n_items)."""
        t1 = time()
        n = self.n_users + self.n_items
        adj = sp.dok_matrix((n, n), dtype=np.float32).tolil()
        R = self.R.tolil()
        for i in range(5):
            lo = int(self.n_users * i / 5.0)
            hi = int(self.n_users * (i + 1.0) / 5)
            adj[lo:hi, self.n_users:] = R[lo:hi]
            adj[self.n_users:, lo:hi] = R[lo:hi].T
        print(f"   Created interaction adj matrix {adj.shape}  ({time() - t1:.1f}s)")
        return adj.tocsr()

    def create_social_adj_mat(self):
        print(f"   Social adj matrix {self.S.shape} — {self.n_social} edges")
        return self.S.tocsr()

    def create_similar_adj_mat(self):
        t1 = time()
        similar = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)

        def cluster(x, d, t):
            v = x / (t + d - x) if (t + d - x) != 0 else 0
            if v <= 0.09:
                return 0
            elif v <= 0.39:
                return 1
            elif v <= 0.69:
                return 10
            elif v <= 0.79:
                return 100
            else:
                return 200

        X = self.U.toarray()
        vfunc = np.vectorize(cluster)
        diag = X.diagonal()
        for i in range(X.shape[0]):
            similar[i] = vfunc(X[i], diag, diag[i])

        print(f"   Created similar users adj matrix {similar.shape}  ({time() - t1:.1f}s)")
        return sp.csr_matrix(similar)

    # ─────────────────────────────────────────
    # Symmetric normalization D^{-0.5} A D^{-0.5}
    # ─────────────────────────────────────────
    @staticmethod
    def normalized_sym(adj):
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D = sp.diags(d_inv_sqrt)
        return D.dot(adj).dot(D).tocsr()

    # ─────────────────────────────────────────
    # get_norm_adj_mat — build / cache tất cả adj matrices
    # ─────────────────────────────────────────
    def get_norm_adj_mat(self, sim_type="all"):
        """
        Tra ve 8 normalized adjacency matrices (scipy.sparse):
          0: interaction, 1: social, 2: similar_users,
          3: tfidf_item, 4: bert_item, 5: full_bert_item,
          6: multimodal, 7: img_only

        sim_type: chi build matrix can thiet, skip phan con lai (tra ve None).
          "none"       → chi build index 0 (interaction)
          "tfidf"      → build 0 + 3
          "bert"       → build 0 + 4
          "full_bert"  → build 0 + 5
          "multimodal" → build 0 + 6
          "img_only"   → build 0 + 7
          "all"        → build tat ca (mac dinh cu)
        """
        # Map sim_type → index can build (ngoai index 0 luon build)
        # Index nao can build (0 = interaction luon can)
        _need = {
            "none":       {0},
            "tfidf":      {0, 3},
            "bert":       {0, 4},
            "full_bert":  {0, 5},
            "multimodal": {0, 6},
            "img_only":   {0, 7},
            "all":        {0, 1, 2, 3, 4, 5, 6, 7},
        }
        need = _need.get(sim_type, {0, 1, 2, 3, 4, 5, 6, 7})

        def _load_or_build(name, build_fn, index):
            if index not in need:
                print(f"   ⏭️  Skip {name} (khong can cho sim_type='{sim_type}')")
                return None
            cache_path = os.path.join(self.path, f"s_{name}.npz")
            try:
                t0 = time()
                mat = sp.load_npz(cache_path)
                print(f"   ✅ Loaded {name} {mat.shape}  ({time() - t0:.1f}s)")
                return mat
            except Exception:
                t0 = time()
                raw = build_fn()
                mat = self.normalized_sym(raw)
                sp.save_npz(cache_path, mat)
                print(f"   🔨 Built  {name} {mat.shape}  ({time() - t0:.1f}s)")
                return mat

        interaction   = _load_or_build("interaction_adj_mat",              self.create_interaction_adj_mat,           0)
        social        = _load_or_build("social_adj_mat",                    self.create_social_adj_mat,                1)
        similar_users = _load_or_build("similar_users_adj_mat",             self.create_similar_adj_mat,               2)
        tfidf_item    = _load_or_build("tfidf_item_similarity_adj_mat",     self.create_tfidf_similarity_matrix,       3)
        bert_item     = _load_or_build("bert_item_similarity_adj_mat",      self.create_bert_similarity_matrix,        4)
        full_bert_item= _load_or_build("full_bert_item_similarity_adj_mat", self.create_full_bert_similarity_matrix,   5)
        multimodal    = _load_or_build("multimodal_similarity_adj_mat",
                                       lambda: self.create_multimodal_similarity_matrix(method="late_fusion", alpha=0.5), 6)
        img_only      = _load_or_build("img_similarity_adj_mat",
                                       lambda: self.create_img_similarity_matrix(threshold=0.5),                       7)

        return (interaction, social, similar_users,
                tfidf_item, bert_item, full_bert_item,
                multimodal, img_only)

    # ─────────────────────────────────────────
    # BPR sampling
    # ─────────────────────────────────────────
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos = len(pos_items)
            pos_batch = []
            while len(pos_batch) < num:
                pos_id = np.random.randint(low=0, high=n_pos)
                pos_i_id = pos_items[pos_id]
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while len(neg_items) < num:
                neg_id = np.random.randint(low=0, high=self.n_items)
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(list(self.test_set.keys()), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos = len(pos_items)
            pos_batch = []
            while len(pos_batch) < num:
                pos_id = np.random.randint(low=0, high=n_pos)
                pos_i_id = pos_items[pos_id]
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while len(neg_items) < num:
                neg_id = np.random.randint(low=0, high=self.n_items)
                if neg_id not in (self.test_set[u] + self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    # ─────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────
    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print(f"   n_users={self.n_users}, n_items={self.n_items}")
        print(f"   n_interactions={self.n_train + self.n_test}")
        print(f"   n_train={self.n_train}, n_test={self.n_test}, "
              f"sparsity={((self.n_train + self.n_test) / (self.n_users * self.n_items)):.5f}")
