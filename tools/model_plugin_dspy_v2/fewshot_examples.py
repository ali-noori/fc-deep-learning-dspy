"""Few-shot prompt/code pairs shown to the language model (data only)."""

from __future__ import annotations

FEWSHOT_PAIRS: list[tuple[str, str]] = [
    # 1.1
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
    # 1.2(Conversational Style for Architecture 1.1)
    (
        """I need a PyTorch class for a CNN that takes 28x28 images. 
        The input channels are given by `in_features` and it outputs `n_classes` categories. 
        First, pass the input through a 2D convolution with 10 output channels and a 5x5 kernel. 
        Then apply a 2x2 max pool and a ReLU. 
        After that, add another 2D conv layer that outputs 20 channels, also with a 5x5 kernel. 
        Apply a 2D dropout, then another 2x2 max pooling and ReLU. 
        Flatten the output, which should be exactly 320 units. 
        Feed this into a fully connected layer with 50 units and a ReLU, followed by a standard dropout. 
        The final linear layer should map to the number of classes. Finish the forward pass with a log softmax.""",
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
        return F.log_softmax(x, dim=1)"""
    ),
    # 2.1
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
    # 2.2 (Conversational Style for Architecture 2.1)
    (
        """Can you help me set up a PyTorch CNN for image classification on 28x28 images? The input will have `in_features` channels, and I need it to output unnormalized logits for `n_classes`. 

Here is the flow I want:
Start off with a 3x3 convolution that maps to 20 channels with a padding of 1. Pass that through a 2x2 max pool and a ReLU. 
For the next block, use another 3x3 conv layer with padding 1 that bumps the channels up to 50, followed by another 2x2 max pool and a ReLU.

After the convolutions, use an adaptive average pool to squeeze the spatial dimensions down to exactly 4x4. Flatten the result—since we have 50 channels, that should give us a flat vector of 800 features. 
Feed that vector into a dense layer with 128 units and a ReLU. Apply a 20% dropout right after that. Finally, add one last linear layer to map to the `n_classes`. Please make sure not to include any softmax or activation at the very end!""",
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
        return x"""
    ),
    # 3.1
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
    # 3.2 (Conversational Style for Architecture 3.1)
    (
        """I need a PyTorch class for a CNN that classifies 28x28 images. The input will have `in_features` channels, and I want to output probabilities for `n_classes` using a log softmax function.

For the feature extractors: start with a 3x3 convolution that outputs 20 channels with a padding of 1. Pass that through a 2x2 max pooling and a ReLU. The second block should be another 3x3 conv with padding 1 that increases the channels to 50, followed by another 2x2 max pool and ReLU. For the third block, use a larger 5x5 kernel with padding 1 to bump the channels up to 100, then do a 2x2 max pool and ReLU again.

After the convolutions, apply an adaptive average pooling layer to squeeze the feature maps down to exactly 4x4. Flatten the result—since we have 100 channels, this should give us a flat vector of 1600 features. Feed that into a fully connected layer with 256 units and a ReLU activation. Add a 20% dropout (0.2) to prevent overfitting, and finish with a linear layer mapping to the number of classes. Don't forget the log softmax along dimension 1 at the end!""",
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
        return F.log_softmax(x, dim=1)"""
    ),
    # 4.1
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
    # 4.2 (Conversational Style for Architecture 4.1)
    (
        """Hey, I need to build a slightly deeper CNN for image classification on 28x28 images. It should take an image with `in_features` channels and output log probabilities for `n_classes` categories.

Let's organize the feature extraction into four blocks. 
For the first block, use a 3x3 convolution with a padding of 1 to output 32 channels. Follow this immediately with a 2D batch normalization, a ReLU activation, and a 2x2 max pool. 
In the second block, do another 3x3 conv with padding 1 to double the channels to 64, followed by batch norm, ReLU, and a 2x2 max pool. 
The third block is the same pattern: a 3x3 conv with padding 1 to reach 128 channels, batch norm, ReLU, and a 2x2 max pool. 
For the fourth and final convolutional block, use a 3x3 conv with padding 1 to bump the channels to 256, apply batch norm and ReLU, but instead of max pooling, use an adaptive average pool to force the spatial dimensions to exactly 2x2.

After that, flatten the tensor. Since we have 256 channels at a 2x2 size, this gives us a flat vector of exactly 1024 features. Feed this 1024-dimensional vector into a dense layer with 512 neurons and a ReLU. Add a dropout layer with a 40% probability (0.4), and finish with a linear layer that projects down to `n_classes`. Return the log softmax of the output along dimension 1.""",
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
        return F.log_softmax(x, dim=1)"""
    ),
    # 5.1
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
    # 5.2 (Conversational Style for Architecture 5.1)
    (
        """I need a standard feedforward neural network (MLP) for a multiclass classification task. The input dimension is `in_features` and it should output raw logits for `n_classes`. 

Start by feeding the input into a dense linear layer with 64 units. Apply a SiLU activation function, followed by a 20% dropout (0.2). Next, project those 64 features down to another linear layer with 32 units, again applying a SiLU activation and a 20% dropout. 

Finally, pass that through a last linear layer that maps down to the number of classes. Please return the unnormalized output directly—do not apply a softmax at the end!""",
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
        return x"""
    ),
    # 6.1
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
    # 6.2 (Conversational Style for Architecture 6.1)
    (
        """Can you build a simple MLP classifier for me? It needs to take `in_features` as the input dimension and output raw predictions for `n_classes` without any softmax activation at the end.

For the hidden layer, map the inputs to a dense layer with 128 units. Immediately run that through a 1D batch normalization, apply a ReLU activation, and then add a 25% dropout (0.25) to help with overfitting. Finally, just project those 128 features directly down to the number of classes using a linear layer.""",
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
        return x"""
    ),
    
]
