import torch.nn as nn
from model.blocks.mha import MultiHeadAttention
from model.blocks.feed_forward import FeedForward
from model.blocks.layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mha_dropout, shortcut_dropout,hidden_dim, max_length, qkv_bias=False):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=max_length,
            num_heads=num_heads,
            mha_dropout=mha_dropout,
            qkv_bias=qkv_bias
        )

        self.ff = FeedForward(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim
        )

        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

        self.drop_shortcut = nn.Dropout(shortcut_dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = shortcut + self.drop_shortcut(x)

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        return x