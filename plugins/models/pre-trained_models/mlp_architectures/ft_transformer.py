"""FT-Transformer (Feature Tokenizer + Transformer) for tabular data.

Architecture-only, trained from scratch on the target dataset — unlike the CNN
backbones under plugins/models/pre-trained_models/cnn_architectures, there is
no generic tabular equivalent of ImageNet to pretrain on, since a tabular
model's input layer is wired directly to one dataset's specific columns.

Same interface as the other plugins/models files: class Model(nn.Module) with
__init__(self, n_classes, in_features) and forward(self, x), where x is shaped
(batch, in_features). Each scalar feature is projected into its own token
embedding, a learnable [CLS] token is prepended, the sequence is passed through
a standard Transformer encoder, and the [CLS] output is used for classification.
"""

import math

import torch
import torch.nn as nn


class _FeatureTokenizer(nn.Module):
    """Projects each scalar feature into its own d_token-dim embedding."""

    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (batch, n_features) -> tokens: (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class Model(nn.Module):
    D_TOKEN = 32
    N_HEADS = 4
    N_LAYERS = 3
    DIM_FEEDFORWARD = 64
    DROPOUT = 0.1

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.tokenizer = _FeatureTokenizer(in_features, self.D_TOKEN)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.D_TOKEN))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_TOKEN,
            nhead=self.N_HEADS,
            dim_feedforward=self.DIM_FEEDFORWARD,
            dropout=self.DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N_LAYERS)

        self.head = nn.Sequential(
            nn.LayerNorm(self.D_TOKEN),
            nn.ReLU(),
            nn.Linear(self.D_TOKEN, n_classes),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]
        return self.head(cls_out)
