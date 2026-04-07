import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet

transfer_type = "full"   # "freeze" | "partial" | "full"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    return (images - IMAGE_MEAN) / IMAGE_STD


def load_checkpoint(path: str, map_location: torch.device):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_encoder_weights(encoder: nn.Module, state_dict: dict) -> None:
    encoder_state = encoder.state_dict()
    filtered = {
        key[len("encoder."):]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
        and key[len("encoder."):] in encoder_state
        and encoder_state[key[len("encoder."):]].shape == value.shape
    }
    encoder.load_state_dict(filtered, strict=False)


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multiclass_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    intersection = (probs * target_onehot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def train():
    seed_everything(42)

    # -----------------------------
    # DATASET
    # -----------------------------
    trainval_dataset = OxfordIIITPetDataset("data", split="trainval")
    test_dataset = OxfordIIITPetDataset("data", split="test")

    generator = torch.Generator().manual_seed(42)

    train_size = int(0.9 * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size

    train_dataset, val_dataset = random_split(
        trainval_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------------
    # MODEL
    # -----------------------------
    model = VGG11UNet().to(device)

    checkpoint = load_checkpoint("checkpoints/classifier.pth", map_location=device)
    load_encoder_weights(model.encoder, checkpoint)

    # -----------------------------
    # TRANSFER LEARNING STRATEGY
    # -----------------------------
    if transfer_type == "freeze":
        for param in model.encoder.parameters():
            param.requires_grad = False

    elif transfer_type == "partial":
        for param in model.encoder.parameters():
            param.requires_grad = False

        # unfreeze last block
        for param in model.encoder.block5.parameters():
            param.requires_grad = True

    elif transfer_type == "full":
        for param in model.encoder.parameters():
            param.requires_grad = True

    # -----------------------------
    # LOSS
    # -----------------------------
    # Background-heavy masks: downweight background to improve macro-dice.
    criterion_ce = nn.CrossEntropyLoss(
        weight=torch.tensor([0.3, 1.0, 1.0], device=device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = 25
    best_loss = float("inf")

    # -----------------------------
    # W&B
    # -----------------------------
    wandb.init(
        project="DA6401_A2_Multitask",
        name=f"transfer_{transfer_type}",
        tags=["transfer_learning", transfer_type],
        config={
            "transfer_type": transfer_type,
            "epochs": epochs,
            "lr": 2e-4
        }
    )

    for epoch in range(epochs):

        # TRAIN
        model.train()
        train_loss = 0

        for batch in train_loader:

            images = normalize_images(batch["image"].to(device))
            masks = batch["mask"].long().to(device)

            # Keep image/mask aligned for segmentation augmentation.
            if torch.rand(1).item() < 0.5:
                images = torch.flip(images, dims=[3])
                masks = torch.flip(masks, dims=[2])

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(images)
                loss_ce = criterion_ce(preds, masks)
                loss_dice = multiclass_dice_loss(preds, masks, num_classes=3)
                loss = 0.5 * loss_ce + 1.5 * loss_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():

            for batch in val_loader:

                images = normalize_images(batch["image"].to(device))
                masks = batch["mask"].long().to(device)

                preds = model(images)
                loss_ce = criterion_ce(preds, masks)
                loss_dice = multiclass_dice_loss(preds, masks, num_classes=3)
                loss = 0.5 * loss_ce + 1.5 * loss_dice

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Train Loss:", train_loss)
        print("Val Loss:", val_loss)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": val_loss,
                },
                "checkpoints/unet.pth"
            )

    wandb.finish()


if __name__ == "__main__":
    train()