"""Simplified TabNet: attention-based sequential feature selection for tabular data.

Architecture-only, trained from scratch on the target dataset — unlike the CNN
backbones under plugins/models/pre-trained_models/cnn_architectures, there is
no generic tabular equivalent of ImageNet to pretrain on, since a tabular
model's input layer is wired directly to one dataset's specific columns.

Same interface as the other plugins/models files: class Model(nn.Module) with
__init__(self, n_classes, in_features) and forward(self, x), where x is shaped
(batch, in_features).

Note: real TabNet uses sparsemax for genuinely sparse feature masks; this
implementation uses softmax instead to stay dependency-free (pure PyTorch,
no extra packages beyond what's already in requirements.txt).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GLUBlock(nn.Module):
    """Linear -> BatchNorm -> Gated Linear Unit (GLU)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim * 2)

    def forward(self, x):
        x = self.bn(self.fc(x))
        return F.glu(x, dim=-1)


class _FeatureTransformer(nn.Module):
    """Two stacked GLU blocks, with a residual connection on the second block."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.block1 = _GLUBlock(in_dim, hidden_dim)
        self.block2 = _GLUBlock(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.block1(x)
        x = (self.block2(x) + x) * (0.5 ** 0.5)
        return x


class _AttentiveTransformer(nn.Module):
    """Produces a soft feature-selection mask, scaled by the running usage prior."""

    def __init__(self, hidden_dim, n_features):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_features)
        self.bn = nn.BatchNorm1d(n_features)

    def forward(self, feat, prior):
        scores = self.bn(self.fc(feat)) * prior
        return F.softmax(scores, dim=-1)


class Model(nn.Module):
    N_STEPS = 3
    HIDDEN_DIM = 32
    GAMMA = 1.5

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.input_bn = nn.BatchNorm1d(in_features)
        self.shared_transformer = _FeatureTransformer(in_features, self.HIDDEN_DIM)
        self.step_transformers = nn.ModuleList(
            [_FeatureTransformer(self.HIDDEN_DIM, self.HIDDEN_DIM) for _ in range(self.N_STEPS)]
        )
        self.attentive_transformers = nn.ModuleList(
            [_AttentiveTransformer(self.HIDDEN_DIM, in_features) for _ in range(self.N_STEPS)]
        )
        self.final_fc = nn.Linear(self.HIDDEN_DIM, n_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.input_bn(x)

        prior = torch.ones_like(x)
        feat = self.shared_transformer(x)
        agg_out = torch.zeros(x.size(0), self.HIDDEN_DIM, device=x.device, dtype=x.dtype)

        for step in range(self.N_STEPS):
            mask = self.attentive_transformers[step](feat, prior)
            prior = prior * (self.GAMMA - mask)
            masked_x = mask * x
            feat = self.step_transformers[step](self.shared_transformer(masked_x))
            agg_out = agg_out + F.relu(feat)

        return self.final_fc(agg_out)
