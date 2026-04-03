"""Unified multi-task model
"""

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


def _load_checkpoint_state(checkpoint_path: str) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _load_prefixed_weights(module: nn.Module, state_dict: dict, prefix: str) -> None:
    filtered_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    module_state = module.state_dict()
    compatible_state = {
        key: value
        for key, value in filtered_state.items()
        if key in module_state and module_state[key].shape == value.shape
    }
    module.load_state_dict(compatible_state, strict=False)


def _load_compatible_state(module: nn.Module, state_dict: dict) -> None:
    module_state = module.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in module_state and module_state[key].shape == value.shape
    }
    module.load_state_dict(compatible_state, strict=False)


class _ClassificationHead(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class _LocalizationHead(nn.Module):
    def __init__(self, image_size: int = 224):
        super().__init__()
        self.image_size = float(image_size)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.regressor(x)) * self.image_size


class _SegmentationDecoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.up5 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, bottleneck: torch.Tensor, features: dict) -> torch.Tensor:
        f1 = features["f1"]
        f2 = features["f2"]
        f3 = features["f3"]
        f4 = features["f4"]
        f5 = features["f5"]

        x = self.up5(bottleneck)
        x = torch.cat([x, f5], dim=1)
        x = self.dec5(x)

        x = self.up4(x)
        x = torch.cat([x, f4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, f3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)

        return self.final(x)


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "checkpoints/classifier.pth", localizer_path: str = "checkpoints/localizer.pth", unet_path: str = "checkpoints/unet.pth"):
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

        self.encoder = VGG11Encoder(in_channels)
        self.classifier_head = _ClassificationHead(num_breeds)
        self.localizer_head = _LocalizationHead()
        self.segmentation_head = _SegmentationDecoder(seg_classes)

        cls_state = _load_checkpoint_state(classifier_path)
        loc_state = _load_checkpoint_state(localizer_path)
        unet_state = _load_checkpoint_state(unet_path)

        _load_prefixed_weights(self.encoder, cls_state, "encoder.")
        _load_prefixed_weights(self.classifier_head.head, cls_state, "classifier.")
        _load_prefixed_weights(self.localizer_head.regressor, loc_state, "regressor.")

        segmentation_state = {
            key: value
            for key, value in unet_state.items()
            if not key.startswith("encoder.")
        }
        _load_compatible_state(self.segmentation_head, segmentation_state)

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
        bottleneck, features = self.encoder(x, return_features=True)

        cls_logits = self.classifier_head(bottleneck)
        bbox = self.localizer_head(bottleneck)
        seg_logits = self.segmentation_head(bottleneck, features)

        return {
            "classification": cls_logits,
            "localization": bbox,
            "segmentation": seg_logits,
        }


