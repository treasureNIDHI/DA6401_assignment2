"""Unified multi-task model
"""

import torch
import torch.nn as nn

from .classification import VGG11Classifier as Classifier
from .localization import VGG11Localizer as Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

        # load trained models
        self.classifier = Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer = Localizer(in_channels=in_channels)
        self.unet = VGG11UNet()

        # load weights
        cls_ckpt = torch.load(classifier_path, map_location="cpu")
        self.classifier.load_state_dict(cls_ckpt["state_dict"])

        loc_ckpt = torch.load(localizer_path, map_location="cpu")
        self.localizer.load_state_dict(loc_ckpt["state_dict"])

        unet_ckpt = torch.load(unet_path, map_location="cpu")
        self.unet.load_state_dict(unet_ckpt["state_dict"])

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # TODO: Implement forward pass.
        # classification
        cls_logits = self.classifier(x)

        # localization
        bbox = self.localizer(x)

        # segmentation
        seg_logits = self.unet(x)

        return {
            "classification": cls_logits,
            "localization": bbox,
            "segmentation": seg_logits,
        }
        raise NotImplementedError("Implement MultiTaskPerceptionModel.forward")


