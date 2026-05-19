"""
models/freedom.py — FREEDOM: Freezing and Denoising for Multimodal Recommendation
=====================================================================================
Paper: "Freedom: Freezing and Denoising Graph Structures for Multimodal Recommendation"
       Zhou et al., ACM MM 2023

sim_type variants:
  img_only             → kNN graph + content từ image only
  tfidf                → kNN graph + content từ text only
  multimodal           → fuse image + text (late fusion)
  multimodal_attention → fuse image + text (attention fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul


class _AttentionFusion(nn.Module):
    """Weighted attention fusion: concat → Linear → d."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, img: Tensor, txt: Tensor) -> Tensor:
        return self.fc(torch.cat([img, txt], dim=1))


def build_knn_item_graph(
    image_feats: "Tensor | None",
    text_feats: "Tensor | None",
    sim_type: str,
    k: int = 10,
    device: torch.device = None,
) -> SparseTensor:
    """
    Xây dựng item-item kNN graph từ modal features theo sim_type.
    cosine sim → top-k neighbors → symmetric → row-normalize.
    """
    assert sim_type in ("img_only", "tfidf", "multimodal", "multimodal_attention")

    if device is None:
        device = (image_feats if image_feats is not None else text_feats).device

    # ── Chọn features để build graph ──
    if sim_type == "img_only":
        feats = F.normalize(image_feats.float().to(device), dim=1)
    elif sim_type == "tfidf":
        feats = F.normalize(text_feats.float().to(device), dim=1)
    else:
        # multimodal / multimodal_attention: concat normalized features
        # (handles dim mismatch, e.g. CLIP=512 vs BERT=768)
        img = F.normalize(image_feats.float().to(device), dim=1)
        txt = F.normalize(text_feats.float().to(device), dim=1)
        feats = F.normalize(torch.cat([img, txt], dim=1), dim=1)

    n = feats.shape[0]
    batch = 256
    rows, cols, vals = [], [], []

    for i in range(0, n, batch):
        chunk = feats[i : i + batch]
        sim = torch.mm(chunk, feats.t())
        sim[:, i : i + chunk.shape[0]].fill_diagonal_(-1.0)

        topk_vals, topk_idx = sim.topk(k, dim=1)
        src = torch.arange(i, i + chunk.shape[0], device=device).unsqueeze(1).expand_as(topk_idx)
        rows.append(src.reshape(-1))
        cols.append(topk_idx.reshape(-1))
        vals.append(topk_vals.reshape(-1))

    rows = torch.cat(rows)
    cols = torch.cat(cols)
    vals = torch.cat(vals)

    # Symmetrize
    rows_sym = torch.cat([rows, cols])
    cols_sym = torch.cat([cols, rows])
    vals_sym = torch.cat([vals, vals])

    adj = SparseTensor(
        row=rows_sym, col=cols_sym, value=vals_sym,
        sparse_sizes=(n, n),
    ).coalesce("sum")

    deg = adj.sum(dim=1).clamp(min=1.0)
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
        image_feats: "Tensor | None",     # None nếu sim_type=tfidf
        text_feats: "Tensor | None",      # None nếu sim_type=img_only
        embedding_dim: int = 64,
        n_layers: int = 2,
        decay: float = 1e-4,
        dropout: float = 0.0,
        knn_k: int = 10,
        cl_weight: float = 0.1,
        cl_temp: float = 0.2,
        sim_type: str = "multimodal",     # img_only | tfidf | multimodal | multimodal_attention
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
        self.sim_type = sim_type

        assert sim_type in ("img_only", "tfidf", "multimodal", "multimodal_attention")
        use_image = sim_type in ("img_only", "multimodal", "multimodal_attention")
        use_text  = sim_type in ("tfidf",    "multimodal", "multimodal_attention")

        # ── ID embeddings ──
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # ── Modal projectors ──
        self.image_projector = None
        self.text_projector  = None

        if use_image:
            assert image_feats is not None
            self.image_projector = nn.Linear(image_feats.shape[1], embedding_dim)
            self.register_buffer("image_feats", image_feats.float())

        if use_text:
            assert text_feats is not None
            self.text_projector = nn.Linear(text_feats.shape[1], embedding_dim)
            self.register_buffer("text_feats", text_feats.float())

        # ── Attention fusion ──
        self.attention_fusion = None
        if sim_type == "multimodal_attention":
            self.attention_fusion = _AttentionFusion(embedding_dim)

        # ── Frozen item-item graph ──
        device = (image_feats if image_feats is not None else text_feats).device
        item_graph = build_knn_item_graph(image_feats, text_feats, sim_type=sim_type, k=knn_k, device=device)
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
    # Modal fusion — chọn theo sim_type
    # ─────────────────────────────────────────
    def _modal_emb(self) -> Tensor:
        if self.sim_type == "img_only":
            return self.image_projector(self.image_feats)

        if self.sim_type == "tfidf":
            return self.text_projector(self.text_feats)

        img = self.image_projector(self.image_feats)
        txt = self.text_projector(self.text_feats)

        if self.sim_type == "multimodal_attention":
            return self.attention_fusion(img, txt)

        return (img + txt) / 2.0   # multimodal late fusion

    # ─────────────────────────────────────────
    # CF propagation
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
        item_emb = self._modal_emb()
        all_embs = [item_emb]
        for _ in range(self.n_layers):
            item_emb = matmul(item_graph, item_emb)
            all_embs.append(item_emb)
        return torch.stack(all_embs, dim=1).mean(dim=1)

    # ─────────────────────────────────────────
    # InfoNCE contrastive loss
    # ─────────────────────────────────────────
    def _infonce(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = torch.mm(z1, z2.t()) / self.cl_temp
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
        item_emb = item_emb_cf + item_emb_content

        # BPR loss
        u_emb = user_emb_cf[users]
        pos_scores = (u_emb * item_emb[pos_items]).sum(dim=1)
        neg_scores = (u_emb * item_emb[neg_items]).sum(dim=1)
        bpr_loss = F.softplus(-(pos_scores - neg_scores)).mean()

        # Regularization
        reg_loss = self.decay * (
            self.user_embedding.weight[users].norm(2).pow(2)
            + self.item_embedding.weight[pos_items].norm(2).pow(2)
            + self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / users.shape[0]

        # InfoNCE: CF view ↔ content view
        cl_loss = self._infonce(item_emb_cf[pos_items], item_emb_content[pos_items])

        loss = bpr_loss + reg_loss + self.cl_weight * cl_loss
        return loss, bpr_loss, reg_loss

    # ─────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        interaction_adj: SparseTensor,
        similarity_adj,   # unused — uniform interface với CombiGCN
        users: Tensor,
    ) -> Tensor:
        user_emb_cf, item_emb_cf = self._cf_propagate(interaction_adj)
        item_emb = item_emb_cf + self._content_propagate()
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
            f"sim_type={self.sim_type}, cl_weight={self.cl_weight})"
        )
