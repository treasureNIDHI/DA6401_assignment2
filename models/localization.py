"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, image_size: int = 224, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            image_size: Fixed image size used to scale normalized box outputs into pixel space.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.image_size = float(image_size)

        # encoder
        self.encoder = VGG11Encoder(in_channels)

        # regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        x = self.encoder(x)
        bbox = torch.sigmoid(self.regressor(x)) * self.image_size

        return bbox
        # TODO: Implement forward pass.
        raise NotImplementedError("Implement VGG11Localizer.forward")

