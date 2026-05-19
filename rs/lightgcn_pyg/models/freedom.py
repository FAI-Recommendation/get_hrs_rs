"""
models/freedom.py — FREEDOM: Freezing and Denoising for Multimodal Recommendation
=====================================================================================
Paper: "Freedom: Freezing and Denoising Graph Structures for Multimodal Recommendation"
       Zhou et al., ACM MM 2023

Ý tưởng chính:
  - Xây dựng item-item graph từ modal features (cosine sim → top-k kNN)
  - "Freeze": item-item graph không được update qua backprop (fixed structure)
  - "Denoise": interaction graph được denoised bằng degree-aware normalization
  - 2 luồng GCN: interaction graph (CF) + item-item graph (content)
  - Fuse CF embedding + content embedding cho items
  - BPR loss + contrastive loss giữa CF view và content view

Input cần có:
  - interaction_adj: normalized bipartite graph (SparseTensor)
  - image_feats:     [n_items, img_dim]
  - text_feats:      [n_items, txt_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul


def build_knn_item_graph(
    image_feats: Tensor,
    text_feats: Tensor,
    k: int = 10,
    device: torch.device = None,
) -> SparseTensor:
    """
    Xây dựng item-item kNN graph từ modal features.
    Cosine sim → top-k neighbors → symmetric → row-normalize.
    """
    if device is None:
        device = image_feats.device

    n = image_feats.shape[0]

    # Fuse image + text features
    img = F.normalize(image_feats.float().to(device), dim=1)
    txt = F.normalize(text_feats.float().to(device), dim=1)
    feats = (img + txt) / 2.0  # [n, d_avg] — late fusion trước khi build graph

    # Tính cosine sim theo batch để tránh OOM
    batch = 256
    rows, cols, vals = [], [], []
    for i in range(0, n, batch):
        chunk = feats[i : i + batch]                   # [b, d]
        sim = torch.mm(chunk, feats.t())               # [b, n]
        sim[:, i : i + chunk.shape[0]].fill_diagonal_(-1.0)  # mask self-loop

        topk_vals, topk_idx = sim.topk(k, dim=1)      # [b, k]

        src = torch.arange(i, i + chunk.shape[0], device=device).unsqueeze(1).expand_as(topk_idx)
        rows.append(src.reshape(-1))
        cols.append(topk_idx.reshape(-1))
        vals.append(topk_vals.reshape(-1))

    rows = torch.cat(rows)
    cols = torch.cat(cols)
    vals = torch.cat(vals)

    # Symmetrize: thêm cả chiều ngược lại
    rows_sym = torch.cat([rows, cols])
    cols_sym = torch.cat([cols, rows])
    vals_sym = torch.cat([vals, vals])

    # Row-normalize
    adj = SparseTensor(
        row=rows_sym, col=cols_sym, value=vals_sym,
        sparse_sizes=(n, n),
    ).coalesce("sum")

    deg = adj.sum(dim=1).clamp(min=1.0)               # [n]
    adj_val = adj.storage.value() / deg[adj.storage.row()]

    return SparseTensor(
        row=adj.storage.row(),
        col=adj.storage.col(),
        value=adj_val.float(),
        sparse_sizes=(n, n),
    )


class FREEDOM(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        image_feats: Tensor,          # [n_items, img_dim]
        text_feats: Tensor,           # [n_items, txt_dim]
        embedding_dim: int = 64,
        n_layers: int = 2,
        decay: float = 1e-4,
        dropout: float = 0.0,
        knn_k: int = 10,              # số neighbors cho item-item graph
        cl_weight: float = 0.1,       # weight của contrastive loss
        cl_temp: float = 0.2,         # temperature cho InfoNCE
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.decay = decay
        self.dropout = dropout
        self.cl_weight = cl_weight
        self.cl_temp = cl_temp

        # ── ID embeddings ──
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # ── Modal projectors ──
        img_dim = image_feats.shape[1]
        txt_dim = text_feats.shape[1]
        self.image_projector = nn.Linear(img_dim, embedding_dim)
        self.text_projector = nn.Linear(txt_dim, embedding_dim)

        self.register_buffer("image_feats", image_feats.float())
        self.register_buffer("text_feats", text_feats.float())

        # ── Frozen item-item graph — built once, không update ──
        device = image_feats.device
        item_graph = build_knn_item_graph(image_feats, text_feats, k=knn_k, device=device)
        # Register as buffer để tự động move sang đúng device
        # SparseTensor không thể register_buffer trực tiếp → lưu COO tensors
        row, col, val = item_graph.coo()
        self.register_buffer("_ig_row", row)
        self.register_buffer("_ig_col", col)
        self.register_buffer("_ig_val", val)
        self._ig_sizes = (n_items, n_items)

    def _get_item_graph(self) -> SparseTensor:
        return SparseTensor(
            row=self._ig_row, col=self._ig_col, value=self._ig_val,
            sparse_sizes=self._ig_sizes,
        )

    # ─────────────────────────────────────────
    # CF propagation trên interaction graph
    # ─────────────────────────────────────────
    def _cf_propagate(self, interaction_adj: SparseTensor) -> tuple[Tensor, Tensor]:
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        ego_emb = torch.cat([user_emb, item_emb], dim=0)
        all_embs = [ego_emb]
        for _ in range(self.n_layers):
            if self.training and self.dropout > 0:
                ego_emb = matmul(self._dropout_sparse(interaction_adj, self.dropout), ego_emb)
            else:
                ego_emb = matmul(interaction_adj, ego_emb)
            all_embs.append(ego_emb)
        final = torch.stack(all_embs, dim=1).mean(dim=1)
        return final[: self.n_users], final[self.n_users :]

    # ─────────────────────────────────────────
    # Content propagation trên frozen item-item graph
    # ─────────────────────────────────────────
    def _content_propagate(self) -> Tensor:
        item_graph = self._get_item_graph()
        img_emb = self.image_projector(self.image_feats)   # [n_items, d]
        txt_emb = self.text_projector(self.text_feats)     # [n_items, d]
        item_emb = (img_emb + txt_emb) / 2.0

        all_embs = [item_emb]
        for _ in range(self.n_layers):
            item_emb = matmul(item_graph, item_emb)
            all_embs.append(item_emb)
        return torch.stack(all_embs, dim=1).mean(dim=1)    # [n_items, d]

    # ─────────────────────────────────────────
    # InfoNCE contrastive loss
    # ─────────────────────────────────────────
    def _infonce(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.mm(z1, z2.t()) / self.cl_temp      # [b, b]
        labels = torch.arange(z1.shape[0], device=z1.device)
        return F.cross_entropy(logits, labels)

    # ─────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────
    def forward(
        self,
        interaction_adj: SparseTensor,
        users: Tensor,
        pos_items: Tensor,
        neg_items: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        user_emb_cf, item_emb_cf = self._cf_propagate(interaction_adj)
        item_emb_content = self._content_propagate()

        # Fuse: CF + content
        item_emb = item_emb_cf + item_emb_content

        # BPR loss
        u_emb = user_emb_cf[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        bpr_loss = F.softplus(-(pos_scores - neg_scores)).mean()

        # Regularization
        reg_loss = self.decay * (
            self.user_embedding.weight[users].norm(2).pow(2)
            + self.item_embedding.weight[pos_items].norm(2).pow(2)
            + self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / users.shape[0]

        # Contrastive: CF view ↔ content view (item-level)
        cl_loss = self._infonce(
            item_emb_cf[pos_items],
            item_emb_content[pos_items],
        )

        loss = bpr_loss + reg_loss + self.cl_weight * cl_loss
        return loss, bpr_loss, reg_loss

    # ─────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        interaction_adj: SparseTensor,
        similarity_adj,   # unused — kept for uniform interface with CombiGCN
        users: Tensor,
    ) -> Tensor:
        user_emb_cf, item_emb_cf = self._cf_propagate(interaction_adj)
        item_emb_content = self._content_propagate()
        item_emb = item_emb_cf + item_emb_content
        return user_emb_cf[users] @ item_emb.t()

    @staticmethod
    def _dropout_sparse(adj: SparseTensor, dropout: float) -> SparseTensor:
        row, col, value = adj.coo()
        mask = torch.rand(value.size(0), device=value.device) > dropout
        row, col, value = row[mask], col[mask], value[mask] / (1.0 - dropout)
        return SparseTensor(row=row, col=col, value=value, sparse_sizes=adj.sizes())

    def __repr__(self):
        return (
            f"FREEDOM(n_users={self.n_users}, n_items={self.n_items}, "
            f"dim={self.embedding_dim}, layers={self.n_layers}, "
            f"knn_k built-in, cl_weight={self.cl_weight})"
        )
