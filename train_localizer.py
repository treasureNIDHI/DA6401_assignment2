import torch
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():

    # -----------------------------
    # DATASET (STRICT ISOLATION)
    # -----------------------------
    trainval_dataset = OxfordIIITPetDataset("data", split="trainval")
    test_dataset = OxfordIIITPetDataset("data", split="test")  # never used for training

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
    checkpoint = torch.load("checkpoints/classifier.pth", map_location=device)
    model.encoder.load_state_dict(checkpoint["state_dict"], strict=False)

    # -----------------------------
    # LOSS
    # -----------------------------
    criterion = IoULoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    best_loss = float("inf")

    for epoch in range(epochs):

        # -----------------------------
        # TRAIN
        # -----------------------------
        model.train()
        train_loss = 0

        for batch in train_loader:

            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device) / 224.0

            preds = model(images)
            preds = torch.sigmoid(preds)

            loss = criterion(preds, bboxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device) / 224.0

                preds = model(images)
                preds = torch.sigmoid(preds)

                loss = criterion(preds, bboxes)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Train Loss:", train_loss)
        print("Val Loss:", val_loss)

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