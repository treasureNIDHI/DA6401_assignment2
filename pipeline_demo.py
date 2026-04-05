import torch
import wandb
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from models.vgg11 import VGG11
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project="DA6401_A2_Multitask",
    name="final_pipeline_showcase",
    tags=["final_pipeline"]
)

# -----------------------------
# LOAD MODELS
# -----------------------------
classifier = VGG11().to(device)
localizer = VGG11Localizer().to(device)
segmenter = VGG11UNet().to(device)

classifier.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device)["state_dict"])
localizer.load_state_dict(torch.load("checkpoints/localizer.pth", map_location=device)["state_dict"], strict=False)
segmenter.load_state_dict(torch.load("checkpoints/unet.pth", map_location=device)["state_dict"])

classifier.eval()
localizer.eval()
segmenter.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# TEST IMAGES
# -----------------------------
paths = [
    "test_images/dog1.jpg",
    "test_images/cat1.jpg",
    "test_images/dog2.jpg"
]

table = wandb.Table(columns=["image"])

for path in paths:

    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # -----------------------------
    # LOCALIZATION
    # -----------------------------
    with torch.no_grad():
        bbox = localizer(tensor)[0].cpu().numpy()

    image = np.array(img.resize((224,224)))

    x,y,w,h = bbox
    x1 = int(x-w/2)
    y1 = int(y-h/2)
    x2 = int(x+w/2)
    y2 = int(y+h/2)

    cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)

    # -----------------------------
    # SEGMENTATION
    # -----------------------------
    with torch.no_grad():
        mask = segmenter(tensor)

    mask = torch.argmax(mask, dim=1)[0].cpu().numpy()
    mask = (mask*255).astype(np.uint8)
    mask = cv2.resize(mask,(224,224))

    color = np.zeros_like(image)
    color[:,:,1] = mask

    image = cv2.addWeighted(image,0.7,color,0.3,0)

    table.add_data(wandb.Image(image))

wandb.log({"Final Pipeline": table})
wandb.finish()