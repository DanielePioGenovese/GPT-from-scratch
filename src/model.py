import torch
import torch.nn as nn

from dataset import TransformerEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, mha_dropout=0.1, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads) == 0, 'd_out must be divisble by num_heads'
        
        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(mha_dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        scores = scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1,2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        context_vec = self.out_proj(context_vec)

        return context_vec
        
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized_x + self.shift

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, ff_dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(ff_dropout)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mha_dropout, shortcut_dropout,hidden_dim, qkv_bias=False):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=emb_dim,
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
    
class GPTModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_length, num_heads, num_layers,shortcut_dropout, mha_dropout, emb_dropout, qkv_bias=False):
        super().__init__()
        self.tok_emb = TransformerEmbedding(
            n_vocab=vocab_size,
            max_length=max_length,
            out_dim=emb_dim,
            emb_dropout=emb_dropout)
        
        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mha_dropout=mha_dropout,
                    shortcut_dropout=shortcut_dropout,
                    hidden_dim=4 * emb_dim,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)

        for block in self.trf_blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
