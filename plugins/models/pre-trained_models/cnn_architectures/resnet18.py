"""Pretrained torchvision ResNet18, ready to fine-tune.

Follows the same interface as the other plugins/models files:
class Model(nn.Module) with __init__(self, n_classes, in_features) and forward(self, x).

- in_features is treated as the number of input image channels (same convention
  as plugins/models/cnn.py), e.g. 1 for grayscale (MNIST), 3 for RGB.
- Non-RGB inputs are adapted to 3 channels with a 1x1 conv so the pretrained
  ImageNet weights can be used as-is.
- Inputs are resized to 224x224 (ResNet18's native ImageNet input size) before
  entering the backbone, so arbitrary input resolutions (e.g. 28x28 MNIST) work.
- The pretrained backbone weights are frozen (requires_grad=False); only the
  new input adapter (if any) and the new final fully-connected head (output
  n_classes) are trained. This keeps training cheap and actually uses the
  pretrained backbone as a fixed feature extractor.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

_WEIGHTS_FILE = "resnet18-f37072fd.pth"


def _empty_resnet18():
    try:
        return models.resnet18(weights=None)
    except TypeError:
        return models.resnet18(pretrained=False)


def _load_local_imagenet(backbone):
    path = Path(__file__).resolve().parent / _WEIGHTS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Local ResNet18 weights not found: {path}"
        )
    backbone.load_state_dict(torch.load(path, map_location="cpu"))


class Model(nn.Module):
    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()

        self.backbone = _empty_resnet18()
        _load_local_imagenet(self.backbone)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.to_rgb = None if in_features == 3 else nn.Conv2d(in_features, 3, kernel_size=1)
        self.resize = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)

        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, n_classes)

    def forward(self, x):
        if self.to_rgb is not None:
            x = self.to_rgb(x)
        x = self.resize(x)
        return self.backbone(x)
