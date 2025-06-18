import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class LinformerSelfAttention(nn.Module):
    """
    A lean version of self-attention: we squish the keys and values
    down to a smaller dimension k before doing dot-product attention.
    This is the core trick from the Linformer paper (Wang et al. 2020):
    "We show that self-attention can be projected onto a low-rank
    subspace, reducing the complexity from O(n^2) to O(nk)."
    Eq. (5) in the paper: Approx softmax(Q K^T / sqrt(d)) V
    but with K, V replaced by E X and F X.
    """
    def __init__(self, embed_dim: int, num_heads: int, k: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # scaling factor 1/sqrt(d)
        self.scale = self.head_dim ** -0.5

        # standard linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # These are the low-rank projection matrices (E for keys, F for values).
        # We learn E, F of shape (k, embed_dim), so seq_len D -> k.
        self.E = nn.Parameter(torch.randn(k, embed_dim) * 0.02)
        self.F = nn.Parameter(torch.randn(k, embed_dim) * 0.02)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: (B, D, C) where D = sequence length, C = embed_dim
        mask: (B, D) with True for positions to keep
        """
        B, D, C = x.shape

        # 1) project inputs to queries, keys, values
        q = self.q_proj(x)  # (B, D, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) split into heads: shape (B, h, D, head_dim)
        def split_heads(t):
            return t.view(B, D, self.num_heads, self.head_dim)\
                    .transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))

        # 3) Linformer trick: project seq_len D down to k for k & v
        #    flatten batch & heads -> (B*h, D, head_dim)
        Bh = B * self.num_heads
        k = k.reshape(Bh, D, self.head_dim)
        v = v.reshape(Bh, D, self.head_dim)
        #    apply E, F: (k, D) x (B*h, D, head_dim) -> (B*h, k, head_dim)
        k = torch.matmul(self.E, k)  # now length k instead of D
        v = torch.matmul(self.F, v)

        # 4) reshape q similarly for matmul: (B*h, D, head_dim)
        q = q.reshape(Bh, D, self.head_dim)

        # 5) compute scaled dot-product attention
        #    eq: A = softmax( Q K^T / sqrt(d) )
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (Bh, D, k)
        if mask is not None:
            # mask: True means keep, False means pad
            # expand mask to (Bh, D, k)
            m = mask.unsqueeze(1).repeat(1, self.num_heads, 1)  # (B, h, D)
            m = m.reshape(Bh, D).unsqueeze(-1).expand(-1, -1, k.size(1))
            attn_scores = attn_scores.masked_fill(~m, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)

        # 6) weighted sum: (Bh, D, k) x (Bh, k, head_dim) -> (Bh, D, head_dim)
        out = torch.matmul(attn, v)
        # 7) reassemble heads -> (B, D, C)
        out = out.view(B, self.num_heads, D, self.head_dim)\
                  .transpose(1, 2)\
                  .reshape(B, D, C)

        # 8) final linear layer
        return self.out_proj(out)


class LinformerEncoderLayer(nn.Module):
    """
    A single transformer encoder block using LinformerSelfAttention.
    Pretty much just like nn.TransformerEncoderLayer, but with our lean attention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # our custom self-attention
        self.self_attn = LinformerSelfAttention(embed_dim, num_heads, k)
        # a small feed-forward network
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        # layer norms & dropouts as usual
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # use ReLU like the original
        self.activation = F.relu

    def forward(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        src: (B, D, C)
        src_key_padding_mask: (B, D), True=keep, False=pad
        """
        x = src
        # 1) self-attention + add & norm
        attn_out = self.self_attn(
            x,
            mask=(~src_key_padding_mask) if src_key_padding_mask is not None else None
        )
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # 2) feed-forward + add & norm
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x
