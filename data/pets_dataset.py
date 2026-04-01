"""Dataset skeleton for Oxford-IIIT Pet.
"""

from torch.utils.data import Dataset
import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET



   
class OxfordIIITPetDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")
        self.xml_dir = os.path.join(root, "annotations", "xmls")

        split_file = os.path.join(root, "annotations", f"{split}.txt")

        with open(split_file) as f:
            self.samples = [line.strip().split()[0] for line in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        name = self.samples[idx]

        img_path = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.masks_dir, name + ".png")
        xml_path = os.path.join(self.xml_dir, name + ".xml")

        # IMAGE
        image = Image.open(img_path).convert("RGB")

        # MASK
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = (mask == 1).astype(np.float32)  # pet = 1
        mask = torch.tensor(mask)

        # BBOX
        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # convert to center format
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        bbox = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

        # LABEL (breed index)
        label = int(name.split("_")[-1]) - 1
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "bbox": bbox,
            "mask": mask
        }