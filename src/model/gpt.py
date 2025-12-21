import torch.nn as nn
from model.blocks import TransformerBlock, LayerNorm
from dataset import TransformerEmbedding

class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        max_length,
        num_heads,
        num_layers,
        shortcut_dropout,
        mha_dropout,
        emb_dropout,
        qkv_bias=False,
    ):
        super().__init__()
        self.tok_emb = TransformerEmbedding(
            n_vocab=vocab_size,
            max_length=max_length,
            out_dim=emb_dim,
            emb_dropout=emb_dropout,
        )

        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mha_dropout=mha_dropout,
                    shortcut_dropout=shortcut_dropout,
                    hidden_dim=4 * emb_dim,
                    max_length=max_length,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

        self.out_head.weight = self.tok_emb.token_embedding.embedding.weight

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)

        for block in self.trf_blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
