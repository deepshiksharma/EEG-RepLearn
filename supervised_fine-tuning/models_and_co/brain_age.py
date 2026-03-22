import torch.nn as nn
from models_and_co.model import The_Encoder


class Brain_Age_Predictor(nn.Module):
    def __init__(self, num_channels=16, embed_dim=128,
                 transformer_depth=2, nhead=4, ff_dim=512, dropout=0.0):
        super().__init__()

        # encoder
        self.encoder = The_Encoder(
            num_channels=num_channels,
            embed_dim=embed_dim,
            transformer_depth=transformer_depth,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.encoder(x, apply_transformer=True)   # (B, 250, 128)

        # global average pooling
        x = x.mean(dim=1)                             # (B, 128)

        x = self.head(x).squeeze(-1)                  # (B,)

        return x
