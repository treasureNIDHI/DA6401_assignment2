import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss


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


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    seed_everything(42)

    # -----------------------------
    # DATASET (STRICT ISOLATION)
    # -----------------------------
    trainval_dataset = OxfordIIITPetDataset("data", split="trainval")
    test_dataset = OxfordIIITPetDataset("data", split="test")  # not used

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
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------------
    # MODEL
    # -----------------------------
    model = VGG11Localizer().to(device)

    # load pretrained encoder
    checkpoint = load_checkpoint("checkpoints/classifier.pth", map_location=device)
    model.encoder.load_state_dict(checkpoint, strict=False)

    # -----------------------------
    # LOSS (MSE + IoU)
    # -----------------------------
    criterion_iou = IoULoss()
    criterion_mse = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = 15
    best_loss = float("inf")

    for epoch in range(epochs):

        # -----------------------------
        # TRAIN
        # -----------------------------
        model.train()
        train_loss = 0

        for batch in train_loader:

            images = normalize_images(batch["image"].to(device))
            bboxes = batch["bbox"].to(device)   # pixel space

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(images)
                preds = torch.clamp(preds, min=0, max=224)

                loss_iou = criterion_iou(preds, bboxes)
                loss_mse = criterion_mse(preds, bboxes)

                loss = loss_mse + loss_iou

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():

            for batch in val_loader:

                images = normalize_images(batch["image"].to(device))
                bboxes = batch["bbox"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(images)
                    preds = torch.clamp(preds, min=0, max=224)
                    loss_iou = criterion_iou(preds, bboxes)
                    loss_mse = criterion_mse(preds, bboxes)

                    loss = loss_mse + loss_iou

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Train Loss:", train_loss)
        print("Val Loss:", val_loss)
        print("LR:", optimizer.param_groups[0]["lr"])

        scheduler.step()

        # -----------------------------
        # SAVE BEST MODEL
        # -----------------------------
        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": val_loss,
                },
                "checkpoints/localizer.pth"
            )

            print("Saved localizer checkpoint")


if __name__ == "__main__":
    train()