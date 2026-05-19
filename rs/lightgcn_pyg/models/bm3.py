"""
models/bm3.py — BM3: Bootstrap Multimodal Contrastive Learning for Recommendation
====================================================================================
Paper: "Bootstrap Latent Representations for Multi-modal Recommendation"
       Zhou et al., WWW 2023

sim_type variants:
  img_only             → chỉ dùng image features
  tfidf                → chỉ dùng text features
  multimodal           → image + text, late fusion (avg)
  multimodal_attention → image + text, weighted attention fusion
"""

import copy

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


class BM3(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        image_feats: "Tensor | None",     # [n_items, img_dim]  — None nếu sim_type=tfidf
        text_feats: "Tensor | None",      # [n_items, txt_dim]  — None nếu sim_type=img_only
        embedding_dim: int = 64,
        n_layers: int = 2,
        decay: float = 1e-4,
        dropout: float = 0.0,
        momentum: float = 0.995,
        cl_weight: float = 0.2,
        sim_type: str = "multimodal",     # img_only | tfidf | multimodal | multimodal_attention
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.decay = decay
        self.dropout = dropout
        self.momentum = momentum
        self.cl_weight = cl_weight
        self.sim_type = sim_type

        assert sim_type in ("img_only", "tfidf", "multimodal", "multimodal_attention"), \
            f"Unknown sim_type for BM3: {sim_type}"
        use_image = sim_type in ("img_only", "multimodal", "multimodal_attention")
        use_text  = sim_type in ("tfidf",    "multimodal", "multimodal_attention")

        # ── ID embeddings ──
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # ── Modal projectors (chỉ tạo cái cần thiết) ──
        self.image_projector = None
        self.text_projector  = None

        if use_image:
            assert image_feats is not None
            img_dim = image_feats.shape[1]
            self.image_projector = nn.Linear(img_dim, embedding_dim)
            self.register_buffer("image_feats", F.normalize(image_feats.float(), dim=1))

        if use_text:
            assert text_feats is not None
            txt_dim = text_feats.shape[1]
            self.text_projector = nn.Linear(txt_dim, embedding_dim)
            self.register_buffer("text_feats", F.normalize(text_feats.float(), dim=1))

        # ── Attention fusion (chỉ cho multimodal_attention) ──
        self.attention_fusion = None
        if sim_type == "multimodal_attention":
            self.attention_fusion = _AttentionFusion(embedding_dim)

        # ── Target (momentum) encoder ──
        self.item_embedding_target = copy.deepcopy(self.item_embedding)
        for p in self.item_embedding_target.parameters():
            p.requires_grad = False

        # ── Predictor head ──
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
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
    # LightGCN propagation
    # ─────────────────────────────────────────
    def _propagate(
        self,
        interaction_adj: SparseTensor,
        user_emb: Tensor,
        item_emb: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
    # EMA update target encoder
    # ─────────────────────────────────────────
    @torch.no_grad()
    def _update_target(self):
        for online, target in zip(
            self.item_embedding.parameters(),
            self.item_embedding_target.parameters(),
        ):
            target.data = self.momentum * target.data + (1.0 - self.momentum) * online.data

    # ─────────────────────────────────────────
    # Bootstrap contrastive loss
    # ─────────────────────────────────────────
    def _bootstrap_loss(self, online: Tensor, target: Tensor) -> Tensor:
        online = F.normalize(self.predictor(online), dim=1)
        target = F.normalize(target.detach(), dim=1)
        return 2.0 - 2.0 * (online * target).sum(dim=1).mean()

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
        user_emb_cf, item_emb_cf = self._propagate(
            interaction_adj,
            self.user_embedding.weight,
            self.item_embedding.weight,
        )
        item_emb_modal = self._modal_emb()
        item_emb = item_emb_cf + item_emb_modal

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

        # Bootstrap contrastive loss
        _, item_emb_target = self._propagate(
            interaction_adj,
            self.user_embedding.weight,
            self.item_embedding_target.weight,
        )
        cl_loss = (
            self._bootstrap_loss(item_emb_cf[pos_items], item_emb_modal[pos_items])
            + self._bootstrap_loss(item_emb_modal[pos_items], item_emb_target[pos_items])
        ) / 2.0

        self._update_target()

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
        user_emb_cf, item_emb_cf = self._propagate(
            interaction_adj,
            self.user_embedding.weight,
            self.item_embedding.weight,
        )
        item_emb = item_emb_cf + self._modal_emb()
        return user_emb_cf[users] @ item_emb.t()

    @staticmethod
    def _dropout_sparse(adj: SparseTensor, dropout: float) -> SparseTensor:
        row, col, value = adj.coo()
        mask = torch.rand(value.size(0), device=value.device) > dropout
        row, col, value = row[mask], col[mask], value[mask] / (1.0 - dropout)
        return SparseTensor(row=row, col=col, value=value, sparse_sizes=adj.sizes())

    def __repr__(self):
        return (
            f"BM3(n_users={self.n_users}, n_items={self.n_items}, "
            f"dim={self.embedding_dim}, layers={self.n_layers}, "
            f"sim_type={self.sim_type}, momentum={self.momentum}, cl_weight={self.cl_weight})"
        )
