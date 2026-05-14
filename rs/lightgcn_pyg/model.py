"""
model.py — CombiGCN (PyG)
==========================
Hỗ trợ 4 chế độ tương ứng 4 file gốc:
  ┌──────────────────────────┬─────────────────────────┬────────────────────┐
  │ File gốc (TF1)          │ similarity_adj          │ Cách dùng PyG      │
  ├──────────────────────────┼─────────────────────────┼────────────────────┤
  │ LightGCN.py              │ None (không dùng)       │ similarity_adj=None│
  │ LightGCN_bert_img.py     │ multimodal (index 6)    │ "multimodal"       │
  │ LightGCN_only_img.py     │ img_only   (index 7)    │ "img_only"         │
  │ LightGCN_tfidf_bert.py   │ tfidf      (index 3)    │ "tfidf"            │
  └──────────────────────────┴─────────────────────────┴────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul


class CombiGCN(nn.Module):
    """
    CombiGCN — hỗ trợ cả LightGCN thuần lẫn dual-graph CombiGCN.

    Khi similarity_adj = None (LightGCN thuần, tương đương hr/LightGCN.py):
        Mỗi layer:  ego_embed = norm_adj @ ego_embed

    Khi similarity_adj != None (CombiGCN, tương đương bert_img / only_img / tfidf):
        Mỗi layer:
            1. interaction_emb = norm_adj @ ego_embed
            2. user_next = interaction_emb[:n_users]
            3. item_interaction = interaction_emb[n_users:]
            4. item_similar = sim_adj @ item_embed
            5. item_next = item_interaction + item_similar   ← fusion (sum)
            6. ego_embed = [user_next ; item_next]

    Final: mean(layer_0, layer_1, ..., layer_K)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        decay: float = 1e-5,
        node_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.decay = decay
        self.node_dropout = node_dropout

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    # ─────────────────────────────────────────
    # Graph propagation
    # ─────────────────────────────────────────
    def get_embedding(
        self,
        interaction_adj: SparseTensor,
        similarity_adj: "SparseTensor | None" = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Graph propagation — tự động chọn LightGCN thuần hoặc CombiGCN.

        Args:
            interaction_adj: (n_users+n_items, n_users+n_items) normalized
            similarity_adj:  (n_items, n_items) normalized, or None

        Returns:
            (user_embeddings, item_embeddings)
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        ego_emb = torch.cat([user_emb, item_emb], dim=0)
        all_embs = [ego_emb]

        use_similarity = similarity_adj is not None

        for _ in range(self.n_layers):
            # 1. Bipartite interaction propagation
            if self.training and self.node_dropout > 0:
                interaction_emb = matmul(
                    self._dropout_sparse(interaction_adj, self.node_dropout),
                    ego_emb,
                )
            else:
                interaction_emb = matmul(interaction_adj, ego_emb)

            if use_similarity:
                # CombiGCN mode (bert_img / only_img / tfidf)
                user_emb_next = interaction_emb[: self.n_users]
                item_emb_interaction = interaction_emb[self.n_users :]

                item_emb_current = ego_emb[self.n_users :]
                item_emb_similar = matmul(similarity_adj, item_emb_current)

                item_emb_next = item_emb_interaction + item_emb_similar

                ego_emb = torch.cat([user_emb_next, item_emb_next], dim=0)
            else:
                # LightGCN thuần mode
                ego_emb = interaction_emb

            all_embs.append(ego_emb)

        all_embs = torch.stack(all_embs, dim=1)
        final_emb = all_embs.mean(dim=1)

        return final_emb[: self.n_users], final_emb[self.n_users :]

    # ─────────────────────────────────────────
    # Forward: BPR training
    # ─────────────────────────────────────────
    def forward(
        self,
        interaction_adj: SparseTensor,
        similarity_adj: "SparseTensor | None",
        users: Tensor,
        pos_items: Tensor,
        neg_items: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        user_emb, item_emb = self.get_embedding(interaction_adj, similarity_adj)

        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        mf_loss = F.softplus(-(pos_scores - neg_scores)).mean()

        u_emb_pre = self.user_embedding.weight[users]
        pos_emb_pre = self.item_embedding.weight[pos_items]
        neg_emb_pre = self.item_embedding.weight[neg_items]

        reg_loss = self.decay * (
            u_emb_pre.norm(2).pow(2)
            + pos_emb_pre.norm(2).pow(2)
            + neg_emb_pre.norm(2).pow(2)
        ) / users.shape[0]

        loss = mf_loss + reg_loss
        return loss, mf_loss, reg_loss

    # ─────────────────────────────────────────
    # Predict: scores cho evaluation
    # ─────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        interaction_adj: SparseTensor,
        similarity_adj: "SparseTensor | None",
        users: Tensor,
    ) -> Tensor:
        user_emb, item_emb = self.get_embedding(interaction_adj, similarity_adj)
        u_emb = user_emb[users]
        scores = u_emb @ item_emb.t()
        return scores

    # ─────────────────────────────────────────
    # Sparse dropout
    # ─────────────────────────────────────────
    @staticmethod
    def _dropout_sparse(adj: SparseTensor, dropout: float) -> SparseTensor:
        row, col, value = adj.coo()
        mask = torch.rand(value.size(0), device=value.device) > dropout
        row = row[mask]
        col = col[mask]
        value = value[mask] / (1.0 - dropout)
        return SparseTensor(
            row=row, col=col, value=value,
            sparse_sizes=adj.sizes(),
        )

    def __repr__(self):
        return (
            f"CombiGCN(n_users={self.n_users}, n_items={self.n_items}, "
            f"dim={self.embedding_dim}, layers={self.n_layers}, "
            f"decay={self.decay}, dropout={self.node_dropout})"
        )


# ─────────────────────────────────────────────
# Helper: convert scipy sparse → SparseTensor
# ─────────────────────────────────────────────
def scipy_to_sparse_tensor(sp_mat, device=None):
    """Convert scipy.sparse matrix → torch_sparse.SparseTensor."""
    coo = sp_mat.tocoo()
    row = torch.from_numpy(coo.row.astype("int64"))
    col = torch.from_numpy(coo.col.astype("int64"))
    val = torch.from_numpy(coo.data.astype("float32"))
    st = SparseTensor(row=row, col=col, value=val, sparse_sizes=coo.shape)
    if device is not None:
        st = st.to(device)
    return st
