"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


def conv_block(in_c, out_c, use_bn):
    layers = [nn.Conv2d(in_c, out_c, 3, padding=1)]

    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))

    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class VGG11Encoder(nn.Module):
    """VGG11-style encoder"""

    def __init__(self, in_channels: int = 3, use_batchnorm: bool = True):
        super().__init__()

        # block1
        self.block1 = conv_block(in_channels, 64, use_batchnorm)
        self.pool1 = nn.MaxPool2d(2,2)

        # block2
        self.block2 = conv_block(64, 128, use_batchnorm)
        self.pool2 = nn.MaxPool2d(2,2)

        # block3
        self.block3 = nn.Sequential(
            conv_block(128, 256, use_batchnorm),
            conv_block(256, 256, use_batchnorm),
        )
        self.pool3 = nn.MaxPool2d(2,2)

        # block4
        self.block4 = nn.Sequential(
            conv_block(256, 512, use_batchnorm),
            conv_block(512, 512, use_batchnorm),
        )
        self.pool4 = nn.MaxPool2d(2,2)

        # block5
        self.block5 = nn.Sequential(
            conv_block(512, 512, use_batchnorm),
            conv_block(512, 512, use_batchnorm),
        )
        self.pool5 = nn.MaxPool2d(2,2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        f1 = self.block1(x)
        p1 = self.pool1(f1)

        f2 = self.block2(p1)
        p2 = self.pool2(f2)

        f3 = self.block3(p2)
        p3 = self.pool3(f3)

        f4 = self.block4(p3)
        p4 = self.pool4(f4)

        f5 = self.block5(p4)
        bottleneck = self.pool5(f5)

        if return_features:
            return bottleneck, {"f1":f1,"f2":f2,"f3":f3,"f4":f4,"f5":f5}

        return bottleneck


def _make_classifier_head(num_classes: int, dropout_p: float):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        CustomDropout(dropout_p),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        CustomDropout(dropout_p),
        nn.Linear(4096, num_classes),
    )


class VGG11(nn.Module):

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True
    ):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels, use_batchnorm)
        self.classifier = _make_classifier_head(num_classes, dropout_p)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)