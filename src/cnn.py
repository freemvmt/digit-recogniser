import torch
import torch.nn as nn
import torch.nn.functional as F


# default hyperparameters (in practice we only adjust scale factor, so we expose it in the training script)
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SCALE_FACTOR = 32
DEFAULT_DENSE_LAYER_MULTIPLIER = 4
DEFAULT_DROPOUT_CONV_P = 0.25
DEFAULT_DROPOUT_FC_P = 0.5


# we build a small convolutional neural network (CNN) appropriate to digit classification
class BaseCNN(nn.Module):
    # we allow for passing in hyperparams at model instantiation
    def __init__(
        self,
        factor: int = DEFAULT_SCALE_FACTOR,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        multiplier: int = DEFAULT_DENSE_LAYER_MULTIPLIER,
        dropout_conv_p: float = DEFAULT_DROPOUT_CONV_P,
        dropout_fc_p: float = DEFAULT_DROPOUT_FC_P,
    ):
        super().__init__()
        # methods for feature extraction
        self.pool = nn.MaxPool2d(2)

        # 1st layer
        self.conv1 = nn.Conv2d(1, factor, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm2d(factor)
        self.dropout_conv1 = nn.Dropout2d(p=dropout_conv_p)

        # 2nd layer
        self.conv2 = nn.Conv2d(factor, factor * 2, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm2d(factor * 2)
        self.dropout_conv2 = nn.Dropout2d(p=dropout_conv_p)

        # methods for classification (fc == fully connected)
        # the input features to fc1 depend on the output size of conv2 after pooling
        self.fc1 = nn.Linear(
            in_features=(factor * 2) * 7 * 7,
            out_features=factor * multiplier,
        )
        self.dropout_fc1 = nn.Dropout(p=dropout_fc_p)
        self.fc2 = nn.Linear(factor * multiplier, 10)

    def forward(self, x):
        # x is a batch of images-as-tensors, with shape (batch_size, 1, 28, 28)
        # NB. dimensions below are based on default scale factor of 32

        # convolutional block 1
        x = self.conv1(x)  # 1x28x28 -> 32x28x28
        x = F.relu(self.bn1(x))
        # order of pool and dropout is inconsequential
        x = self.pool(x)  # 32x28x28 -> 32x14x14
        x = self.dropout_conv1(x)

        # convolutional block 2
        x = self.conv2(x)  # 32x14x14 -> 64x14x14
        x = F.relu(self.bn2(x))
        x = self.pool(x)  # 64x14x14 -> 64x7x7
        x = self.dropout_conv2(x)

        x = torch.flatten(x, 1)  # flatten to (batch_size, 64*7*7)

        # fully connected block 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)

        # output layer spits out 10 logits (1 per digit class)
        return self.fc2(x)
