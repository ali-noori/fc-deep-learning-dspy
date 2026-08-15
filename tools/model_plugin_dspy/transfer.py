"""Transfer-learning backends (torchvision) for model_plugin_dspy."""

from __future__ import annotations

import math
from dataclasses import dataclass

ALLOWED_TRANSFER_MODELS: tuple[str, ...] = (
    "ResNet18",
    "ResNet50",
    "EfficientNet-B0",
    "VGG16",
    "MobileNetV3",
    "MobileNetV3-Small",
    "ViT",
    "SqueezeNet",
)

_ALIAS_TO_CANONICAL: dict[str, str] = {
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "efficientnet-b0": "EfficientNet-B0",
    "efficientnetb0": "EfficientNet-B0",
    "vgg16": "VGG16",
    "mobilenetv3": "MobileNetV3",
    "mobilenet-v3": "MobileNetV3",
    "mobilenetv3small": "MobileNetV3-Small",
    "mobilenetv3-small": "MobileNetV3-Small",
    "mobilenet-v3-small": "MobileNetV3-Small",
    "vit": "ViT",
    "visiontransformer": "ViT",
    "squeezenet": "SqueezeNet",
    "squeezenet1_0": "SqueezeNet",
    "squeezenet1.0": "SqueezeNet",
}


def normalize_transfer_model_name(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "").replace("_", "-")
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    for allowed in ALLOWED_TRANSFER_MODELS:
        if raw.strip().lower() == allowed.lower().replace(" ", ""):
            return allowed
        if raw.strip().lower() == allowed.lower():
            return allowed
    raise ValueError(
        f"Unknown transfer model {raw!r}. Choose one of: {', '.join(ALLOWED_TRANSFER_MODELS)}"
    )


@dataclass
class TransferPluginConfig:
    canonical_name: str
    dataset_size: int
    input_dim: int
    output_dim: int
    input_channels: int
    image_height: int
    image_width: int


def infer_spatial_size(input_dim: int, for_vit: bool) -> tuple[int, int]:
    if for_vit:
        return 224, 224
    if input_dim > 0:
        r = int(math.isqrt(input_dim))
        if r * r == input_dim:
            return r, r
    return 224, 224


def build_transfer_config(
    canonical_name: str,
    dataset_size: int,
    input_dim: int,
    output_dim: int,
    input_channels: int | None,
) -> TransferPluginConfig:
    if output_dim <= 0:
        raise ValueError("transfer-learning JSON must include a positive output_dim (n_classes).")
    if input_dim <= 0:
        raise ValueError(
            "transfer-learning JSON must include a positive input_dim (e.g. 784 for 28x28 flattened)."
        )
    if dataset_size <= 0:
        dataset_size = 1
    ch = 1 if input_channels is None else max(1, min(32, int(input_channels)))
    for_vit = canonical_name == "ViT"
    h, w = infer_spatial_size(input_dim, for_vit=for_vit)
    return TransferPluginConfig(
        canonical_name=canonical_name,
        dataset_size=dataset_size,
        input_dim=input_dim,
        output_dim=output_dim,
        input_channels=ch,
        image_height=h,
        image_width=w,
    )


def render_transfer_model_file(cfg: TransferPluginConfig) -> str:
    name = cfg.canonical_name
    if name in _LIGHTWEIGHT_TRANSFER:
        return _LIGHTWEIGHT_TRANSFER[name](cfg)
    if name == "ResNet18":
        return _tpl_resnet("resnet18", "ResNet18_Weights", cfg)
    if name == "ResNet50":
        return _tpl_resnet("resnet50", "ResNet50_Weights", cfg)
    if name == "VGG16":
        return _tpl_vgg16(cfg)
    if name == "MobileNetV3":
        return _tpl_mobilenet_v3_large(cfg)
    if name == "ViT":
        return _tpl_vit_b16(cfg)
    raise ValueError(f"Unsupported transfer model: {name}")


_LIGHTWEIGHT_HEAD = (
    "        if x.dim() != 4:\n"
    "            raise ValueError('Expected NCHW image tensor')\n"
    "        x = self.to_rgb(x)\n"
    "        x = self.upsample(x)\n"
    "        x = self.backbone(x)\n"
)


def _lightweight_init_and_wrap(cfg: TransferPluginConfig, backbone_block: str, head_block: str, tail_forward: str) -> str:
    comment = (
        f"        # Transfer: {cfg.canonical_name}; metadata dataset_size={cfg.dataset_size}, "
        f"input_dim={cfg.input_dim}; in_features=input channels\n"
    )
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import torchvision.models as models\n\n\n"
        "class Model(nn.Module):\n"
        f'    """Auto-generated transfer model ({cfg.canonical_name})."""\n\n'
        "    def __init__(self, n_classes, in_features):\n"
        "        super(Model, self).__init__()\n"
        f"{comment}"
        "        self.to_rgb = nn.Conv2d(in_features, 3, kernel_size=1)\n"
        "        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)\n"
        f"{backbone_block}\n"
        "        for param in self.backbone.parameters():\n"
        "            param.requires_grad = False\n"
        f"{head_block}\n\n"
        "    def forward(self, x):\n"
        f"{_LIGHTWEIGHT_HEAD}"
        f"{tail_forward}"
    )


def _tpl_mobilenet_v3_small(cfg: TransferPluginConfig) -> str:
    bb = (
        "        try:\n"
        "            w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1\n"
        "            self.backbone = models.mobilenet_v3_small(weights=w)\n"
        "        except Exception:\n"
        "            try:\n"
        "                self.backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')\n"
        "            except Exception:\n"
        "                self.backbone = models.mobilenet_v3_small(pretrained=True)"
    )
    head = (
        "        in_f = self.backbone.classifier[-1].in_features\n"
        "        self.backbone.classifier[-1] = nn.Linear(in_f, n_classes)"
    )
    tail = "        return F.log_softmax(x, dim=1)\n"
    return _lightweight_init_and_wrap(cfg, bb, head, tail)


def _tpl_efficientnet_b0_lightweight(cfg: TransferPluginConfig) -> str:
    bb = (
        "        try:\n"
        "            w = models.EfficientNet_B0_Weights.IMAGENET1K_V1\n"
        "            self.backbone = models.efficientnet_b0(weights=w)\n"
        "        except Exception:\n"
        "            try:\n"
        "                self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')\n"
        "            except Exception:\n"
        "                self.backbone = models.efficientnet_b0(pretrained=True)"
    )
    head = (
        "        in_f = self.backbone.classifier[-1].in_features\n"
        "        self.backbone.classifier[-1] = nn.Linear(in_f, n_classes)"
    )
    tail = "        return F.log_softmax(x, dim=1)\n"
    return _lightweight_init_and_wrap(cfg, bb, head, tail)


def _tpl_squeezenet_lightweight(cfg: TransferPluginConfig) -> str:
    bb = (
        "        try:\n"
        "            w = models.SqueezeNet1_0_Weights.IMAGENET1K_V1\n"
        "            self.backbone = models.squeezenet1_0(weights=w)\n"
        "        except Exception:\n"
        "            try:\n"
        "                self.backbone = models.squeezenet1_0(weights='IMAGENET1K_V1')\n"
        "            except Exception:\n"
        "                self.backbone = models.squeezenet1_0(pretrained=True)"
    )
    head = "        self.backbone.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=1)"
    tail = (
        "        x = torch.flatten(x, start_dim=1)\n"
        "        return F.log_softmax(x, dim=1)\n"
    )
    return _lightweight_init_and_wrap(cfg, bb, head, tail)


_LIGHTWEIGHT_TRANSFER = {
    "MobileNetV3-Small": _tpl_mobilenet_v3_small,
    "EfficientNet-B0": _tpl_efficientnet_b0_lightweight,
    "SqueezeNet": _tpl_squeezenet_lightweight,
}


def _head_comment(cfg: TransferPluginConfig) -> str:
    return (
        f"        # Transfer: {cfg.canonical_name}; "
        f"dataset_size={cfg.dataset_size}, input_dim={cfg.input_dim}; "
        "1ch repeated to 3 when C==1\n"
    )


def _input_adapter() -> str:
    return (
        "        if x.dim() == 2:\n"
        "            raise ValueError('Expected NCHW image tensor for transfer model')\n"
        "        if x.size(1) == 1:\n"
        "            x = x.repeat(1, 3, 1, 1)\n"
        "        elif x.size(1) != 3:\n"
        "            raise ValueError(f'Expected 1 or 3 input channels, got {x.size(1)}')\n"
    )


def _tpl_resnet(fn: str, weights_cls: str, cfg: TransferPluginConfig) -> str:
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torchvision.models as models\n\n\n"
        "class Model(nn.Module):\n"
        '    """Auto-generated transfer model (torchvision ResNet)."""\n\n'
        "    def __init__(self, n_classes, in_features):\n"
        "        super(Model, self).__init__()\n"
        f"{_head_comment(cfg)}"
        "        try:\n"
        f"            w = getattr(models, '{weights_cls}').DEFAULT\n"
        f"            self.backbone = models.{fn}(weights=w)\n"
        "        except Exception:\n"
        f"            self.backbone = models.{fn}(pretrained=True)\n"
        "        nf = self.backbone.fc.in_features\n"
        "        self.backbone.fc = nn.Linear(nf, n_classes)\n\n"
        "    def forward(self, x):\n"
        f"{_input_adapter()}"
        "        return self.backbone(x)\n"
    )


def _tpl_vgg16(cfg: TransferPluginConfig) -> str:
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import torchvision.models as models\n\n\n"
        "class Model(nn.Module):\n"
        '    """Auto-generated transfer model (VGG16)."""\n\n'
        "    def __init__(self, n_classes, in_features):\n"
        "        super(Model, self).__init__()\n"
        f"{_head_comment(cfg)}"
        "        try:\n"
        "            w = models.VGG16_Weights.DEFAULT\n"
        "            self.backbone = models.vgg16(weights=w)\n"
        "        except Exception:\n"
        "            self.backbone = models.vgg16(pretrained=True)\n"
        "        in_f = self.backbone.classifier[6].in_features\n"
        "        self.backbone.classifier[6] = nn.Linear(in_f, n_classes)\n\n"
        "    def forward(self, x):\n"
        f"{_input_adapter()}"
        "        if x.size(-2) < 224 or x.size(-1) < 224:\n"
        "            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)\n"
        "        return self.backbone(x)\n"
    )


def _tpl_mobilenet_v3_large(cfg: TransferPluginConfig) -> str:
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torchvision.models as models\n\n\n"
        "class Model(nn.Module):\n"
        '    """Auto-generated transfer model (MobileNetV3-Large)."""\n\n'
        "    def __init__(self, n_classes, in_features):\n"
        "        super(Model, self).__init__()\n"
        f"{_head_comment(cfg)}"
        "        try:\n"
        "            w = models.MobileNet_V3_Large_Weights.DEFAULT\n"
        "            self.backbone = models.mobilenet_v3_large(weights=w)\n"
        "        except Exception:\n"
        "            self.backbone = models.mobilenet_v3_large(pretrained=True)\n"
        "        in_f = self.backbone.classifier[3].in_features\n"
        "        self.backbone.classifier[3] = nn.Linear(in_f, n_classes)\n\n"
        "    def forward(self, x):\n"
        f"{_input_adapter()}"
        "        return self.backbone(x)\n"
    )


def _tpl_vit_b16(cfg: TransferPluginConfig) -> str:
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torchvision.models as models\n\n\n"
        "class Model(nn.Module):\n"
        '    """Auto-generated transfer model (ViT-B/16)."""\n\n'
        "    def __init__(self, n_classes, in_features):\n"
        "        super(Model, self).__init__()\n"
        f"{_head_comment(cfg)}"
        "        try:\n"
        "            w = models.ViT_B_16_Weights.DEFAULT\n"
        "            self.backbone = models.vit_b_16(weights=w)\n"
        "        except Exception:\n"
        "            self.backbone = models.vit_b_16(pretrained=True)\n"
        "        in_f = self.backbone.heads.head.in_features\n"
        "        self.backbone.heads.head = nn.Linear(in_f, n_classes)\n\n"
        "    def forward(self, x):\n"
        f"{_input_adapter()}"
        "        return self.backbone(x)\n"
    )
