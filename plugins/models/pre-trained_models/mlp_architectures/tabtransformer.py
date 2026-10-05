"""TabTransformer-style architecture for tabular data.

Architecture-only, trained from scratch on the target dataset — unlike the CNN
backbones under plugins/models/pre-trained_models/cnn_architectures, there is
no generic tabular equivalent of ImageNet to pretrain on, since a tabular
model's input layer is wired directly to one dataset's specific columns.

The original TabTransformer paper embeds categorical columns through a
Transformer and concatenates the result with normalized continuous columns
before an MLP head. This plugin interface only exposes n_classes/in_features
(no per-column categorical/continuous metadata), so here every feature is
tokenized and passed through the Transformer, and the contextualized tokens
are concatenated with the normalized raw features (skip connection) before
the MLP head — approximating the paper's design without requiring per-column
type metadata.

Same interface as the other plugins/models files: class Model(nn.Module) with
__init__(self, n_classes, in_features) and forward(self, x), where x is shaped
(batch, in_features).
"""

import math

import torch
import torch.nn as nn


class _ColumnTokenizer(nn.Module):
    """Per-feature learned token embedding for the transformer branch."""

    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class Model(nn.Module):
    D_TOKEN = 32
    N_HEADS = 4
    N_LAYERS = 2
    DIM_FEEDFORWARD = 64
    DROPOUT = 0.1
    MLP_HIDDEN = 64

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.input_bn = nn.BatchNorm1d(in_features)
        self.tokenizer = _ColumnTokenizer(in_features, self.D_TOKEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_TOKEN,
            nhead=self.N_HEADS,
            dim_feedforward=self.DIM_FEEDFORWARD,
            dropout=self.DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N_LAYERS)

        mlp_in = in_features * self.D_TOKEN + in_features
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, self.MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.MLP_HIDDEN, n_classes),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x_norm = self.input_bn(x)
        tokens = self.tokenizer(x_norm)
        encoded = self.encoder(tokens)
        encoded_flat = encoded.reshape(x.size(0), -1)
        combined = torch.cat([encoded_flat, x_norm], dim=1)
        return self.mlp(combined)
