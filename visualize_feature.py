import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from models.vgg11 import VGG11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = VGG11().to(device)

checkpoint = torch.load("checkpoints/classifier.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# -----------------------------
# HOOKS
# -----------------------------
features = {}

def hook_first(module, input, output):
    features["first"] = output.detach()

def hook_last(module, input, output):
    features["last"] = output.detach()

# first conv
model.encoder.block1[0].register_forward_hook(hook_first)

# find LAST conv layer automatically
for layer in reversed(list(model.encoder.modules())):
    if isinstance(layer, torch.nn.Conv2d):
        layer.register_forward_hook(hook_last)
        break

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = Image.open("data/images/Abyssinian_1.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = transform(img).unsqueeze(0).to(device)

# -----------------------------
# FORWARD
# -----------------------------
with torch.no_grad():
    _ = model(img)

first = features["first"][0]
last = features["last"][0]

# -----------------------------
# FIRST LAYER
# -----------------------------
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(first[i].cpu(), cmap="gray")
    plt.axis("off")

plt.suptitle("First Conv Layer Features")
plt.show()

# -----------------------------
# LAST LAYER
# -----------------------------
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(last[i].cpu(), cmap="gray")
    plt.axis("off")

plt.suptitle("Last Conv Layer Features")
plt.show()