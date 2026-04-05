import torch
import wandb
import numpy as np
import cv2

from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="DA6401_A2_Multitask", name="detection_visualization")


# -----------------------------
# LOAD MODEL
# -----------------------------
model = VGG11Localizer().to(device)

checkpoint = torch.load("checkpoints/localizer.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval()


# -----------------------------
# DATASET
# -----------------------------
dataset = OxfordIIITPetDataset("data", split="trainval")
loader = DataLoader(dataset, batch_size=1, shuffle=True)


# -----------------------------
# IoU
# -----------------------------
def compute_iou(box1, box2):

    def to_xyxy(box):
        x, y, w, h = box
        return [
            x - w/2,
            y - h/2,
            x + w/2,
            y + h/2
        ]

    box1 = to_xyxy(box1)
    box2 = to_xyxy(box2)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter / (union + 1e-6)


# -----------------------------
# TABLE
# -----------------------------
table = wandb.Table(columns=["image", "confidence", "IoU"])

count = 0

for batch in loader:

    img = batch["image"].to(device)
    gt = batch["bbox"][0].cpu().numpy()

    with torch.no_grad():
        pred = model(img)[0].cpu().numpy()

    confidence = float(np.mean(pred))
    iou = compute_iou(pred, gt)

    # -----------------------------
    # convert image (FIXED)
    # -----------------------------
    image = img[0].cpu().permute(1,2,0).numpy()
    image = (image * 255).clip(0,255).astype(np.uint8)
    image = np.ascontiguousarray(image)

    # -----------------------------
    # draw GT (green)
    # -----------------------------
    x,y,w_box,h_box = gt.astype(int)

    x1 = int(x - w_box/2)
    y1 = int(y - h_box/2)
    x2 = int(x + w_box/2)
    y2 = int(y + h_box/2)

    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

    # -----------------------------
    # draw pred (red)
    # -----------------------------
    x,y,w_box,h_box = pred.astype(int)

    x1 = int(x - w_box/2)
    y1 = int(y - h_box/2)
    x2 = int(x + w_box/2)
    y2 = int(y + h_box/2)

    cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)

    table.add_data(
        wandb.Image(image),
        confidence,
        iou
    )

    count += 1
    if count == 10:
        break

wandb.log({"Detection Results": table})
wandb.finish()