import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.init(project="DA6401_A2_Multitask", name="segmentation_visualization")
wandb.init(
    project="DA6401_A2_Multitask",
    name="segmentation_visualization",
    tags=["segmentation", "dice_vs_accuracy", "visualization"]
)


# -----------------------------
# LOAD MODEL
# -----------------------------
model = VGG11UNet().to(device)

checkpoint = torch.load("checkpoints/unet.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"], strict=False)

model.eval()


# -----------------------------
# DATASET
# -----------------------------
dataset = OxfordIIITPetDataset("data", split="trainval")
loader = DataLoader(dataset, batch_size=1, shuffle=True)


# -----------------------------
# METRICS
# -----------------------------
def pixel_accuracy(pred, target):
    correct = (pred == target).sum()
    total = torch.numel(pred)
    return (correct / total).item()


def dice_score(pred, target):

    pred = pred == 1
    target = target == 1

    intersection = (pred & target).sum()
    union = pred.sum() + target.sum()

    return (2 * intersection / (union + 1e-6)).item()


# -----------------------------
# TABLE
# -----------------------------
table = wandb.Table(
    columns=["image", "GT", "prediction", "pixel_acc", "dice"]
)

count = 0

for batch in loader:

    img = batch["image"].to(device)
    gt = batch["mask"].to(device)

    with torch.no_grad():
        pred = model(img)
        pred = torch.argmax(pred, dim=1)

    acc = pixel_accuracy(pred, gt)
    dice = dice_score(pred, gt)

    image = img[0].cpu().permute(1,2,0).numpy()
    gt_mask = gt[0].cpu().numpy()
    pred_mask = pred[0].cpu().numpy()

    table.add_data(
        wandb.Image(image),
        wandb.Image(gt_mask),
        wandb.Image(pred_mask),
        acc,
        dice
    )

    count += 1
    if count == 5:
        break

wandb.log({"Segmentation Results": table})
wandb.finish()