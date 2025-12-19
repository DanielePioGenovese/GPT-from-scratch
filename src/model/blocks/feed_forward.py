import torch.nn as nn

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