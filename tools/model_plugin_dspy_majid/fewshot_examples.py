"""Few-shot prompt/code pairs shown to the language model (data only)."""

from __future__ import annotations

FEWSHOT_PAIRS: list[tuple[str, str]] = [
    (
        """Create an CNN architecture for image classification. The dataset contains 28x28 images with in_features channel, and the model should perform multiclass classification with n_classes = 10

Conv2d: in_features->10, kernel=5, no padding
MaxPool2d(2) -> ReLU
Conv2d: 10->20, kernel=5, no padding
Dropout2d
MaxPool2d(2) -> ReLU
Flatten to 320
Linear: 320->50 -> ReLU
Dropout
Linear: 50->n_classes
log_softmax output""",
        """import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = self.act(self.pool(self.conv1(x)))
        x = self.act(self.pool(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)""",
    ),
    (
        """Create an CNN architecture for image classification. The dataset contains 28x28 images with in_features channel, and the model should perform multiclass classification with n_classes = 10

Conv2d: in_features->20, kernel=3, padding=1
MaxPool2d(2) -> ReLU
Conv2d: 20->50, kernel=3, padding=1
MaxPool2d(2) -> ReLU
AdaptiveAvgPool2d(4x4)
Flatten to 800
Linear: 800->128 -> ReLU
Dropout1d(0.2)
Linear: 128->n_classes
no softmax output""",
        """import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(50 * 4 * 4, 128)
        self.drop = nn.Dropout(0.2)
        self.fc_out = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.act(self.pool(self.conv1(x)))
        x = self.act(self.pool(self.conv2(x)))
        x = self.adapt(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc_out(x)
        return x""",
    ),
    (
        """Create an CNN architecture for image classification. The dataset contains 28x28 images with in_features channel, and the model should perform multiclass classification with n_classes = 10

Conv2d: in_features->20, kernel=3, padding=1
MaxPool2d(2) -> ReLU
Conv2d: 20->50, kernel=3, padding=1
MaxPool2d(2) -> ReLU
Conv2d: 50->100, kernel=5, padding=1
MaxPool2d(2) -> ReLU
AdaptiveAvgPool2d(4x4)
Flatten to 1600
Linear: 1600->256 -> ReLU
Dropout1d(0.2)
Linear: 256->n_classes
log_softmax output""",
        """import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(50, 100, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(100 * 4 * 4, 256)
        self.drop = nn.Dropout(0.2)
        self.fc_out = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.act(self.pool(self.conv1(x)))
        x = self.act(self.pool(self.conv2(x)))
        x = self.act(self.pool(self.conv3(x)))
        x = self.adapt(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)""",
    ),
    (
        """Create an CNN architecture for image classification. The dataset contains 28x28 images with in_features channel, and the model should perform multiclass classification with n_classes = 10

Conv2d: in_features->32, kernel=3, padding=1
BatchNorm2d
ReLU
MaxPool2d(2)
Conv2d: 32->64, kernel=3, padding=1
BatchNorm2d
ReLU
MaxPool2d(2)
Conv2d: 64->128, kernel=3, padding=1
BatchNorm2d
ReLU
MaxPool2d(2)
Conv2d: 128->256, kernel=3, padding=1
BatchNorm2d
ReLU
AdaptiveAvgPool2d(2x2)
Flatten to 1024
Linear: 1024->512 -> ReLU
Dropout(0.4)
Linear: 512->n_classes
log_softmax output""",
        """import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.adapt = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.drop = nn.Dropout(0.4)
        self.fc_out = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.act(self.bn4(self.conv4(x)))
        x = self.adapt(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)""",
    ),
    (
        """Create an MLP architecture for multiclass classification with n_classes classes and in_features input features.

Linear: in_features->64 -> SiLU
Dropout(0.2)
Linear: 64->32 -> SiLU
Dropout(0.2)
Linear: 32->n_classes
no softmax output""",
        """import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, n_classes)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.act(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        return x""",
    ),
    (
        """Create an MLP architecture for multiclass classification with n_classes classes and in_features input features.

Linear: in_features->128 -> BatchNorm1d -> ReLU
Dropout(0.25)
Linear: 128->n_classes
no softmax""",
        """import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, n_classes)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.fc_out(x)
        return x""",
    ),
]
