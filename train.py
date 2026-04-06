import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
import os

from data.pets_dataset import OxfordIIITPetDataset
from models.vgg11 import VGG11


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
    # CHANGE ONLY THIS FOR 3 RUNS
    # -----------------------------
    dropout_p = 0.5  # run1: 0.0 | run2: 0.2 | run3: 0.5

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
    model = VGG11(
        dropout_p=dropout_p,
        use_batchnorm=True
    ).to(device)

    # -----------------------------
    # LOSS
    # -----------------------------
    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # OPTIMIZER
    # -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = 10
    
    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")

    # -----------------------------
    # W&B INIT (WITH TAGS)
    # -----------------------------
    wandb.init(
        project="DA6401_A2_Multitask",
        name=f"dropout_{dropout_p}",
        tags=["dropout_experiment", f"dropout_{dropout_p}"],
        config={
            "model": "VGG11",
            "dropout": dropout_p,
            "batchnorm": True,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "batch_size": 16,
            "epochs": epochs
        }
    )

    for epoch in range(epochs):

        # -----------------------------
        # TRAIN
        # -----------------------------
        model.train()
        train_loss = 0

        for batch in train_loader:

            images = normalize_images(batch["image"].to(device))
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
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

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Train Loss:", train_loss)
        print("Val Loss:", val_loss)
        print("Val Acc:", acc)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": acc
        })

        # Save best checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": val_loss,
                },
                "checkpoints/classifier.pth"
            )
            print(f"✓ Saved classifier checkpoint (val_loss={val_loss:.4f})")

        scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    train()