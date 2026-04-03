import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    return (images - IMAGE_MEAN) / IMAGE_STD


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_classification_batch(images: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        images = torch.flip(images, dims=[3])
    return images


def train():
    seed_everything(42)

    # -----------------------------
    # DATASET (STRICT ISOLATION)
    # -----------------------------
    trainval_dataset = OxfordIIITPetDataset("data", split="trainval")
    test_dataset = OxfordIIITPetDataset("data", split="test")  # never used for training

    # deterministic split
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
    model = VGG11Classifier().to(device)

    # -----------------------------
    # LOSS
    # -----------------------------
    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # OPTIMIZER
    # -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = 20
    best_acc = 0

    for epoch in range(epochs):

        # -----------------------------
        # TRAIN
        # -----------------------------
        model.train()
        train_loss = 0

        for batch in train_loader:

            images = batch["image"].to(device)
            images = augment_classification_batch(images)
            images = normalize_images(images)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

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
        correct = 0
        total = 0

        with torch.no_grad():

            for batch in val_loader:

                images = normalize_images(batch["image"].to(device))
                labels = batch["label"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total if total > 0 else 0

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Train Loss:", train_loss / len(train_loader))
        print("Val Loss:", val_loss / len(val_loader))
        print("Val Acc:", acc)
        print("LR:", optimizer.param_groups[0]["lr"])

        scheduler.step()

        # -----------------------------
        # SAVE BEST MODEL
        # -----------------------------
        if acc > best_acc:
            best_acc = acc

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": acc,
                },
                "checkpoints/classifier.pth"
            )

            print("Saved classifier checkpoint")


if __name__ == "__main__":
    train()