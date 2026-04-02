import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms

from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root, split="trainval", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")
        self.xml_dir = os.path.join(root, "annotations", "xmls")

        split_file = os.path.join(root, "annotations", f"{split}.txt")

        with open(split_file) as f:
            # self.samples = [line.strip().split()[0] for line in f.readlines()]
            samples = [line.strip().split()[0] for line in f.readlines()]

            # remove samples without xml
            self.samples = []
            for s in samples:
                xml_path = os.path.join(self.xml_dir, s + ".xml")
                if os.path.exists(xml_path):
                    self.samples.append(s)

        self.to_tensor = transforms.ToTensor()

        # create breed mapping
        breeds = sorted(set("_".join(s.split("_")[:-1]) for s in self.samples))
        self.breed_to_idx = {b: i for i, b in enumerate(breeds)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        name = self.samples[idx]

        img_path = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.masks_dir, name + ".png")
        xml_path = os.path.join(self.xml_dir, name + ".xml")

        # IMAGE
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224,224))
        image = self.to_tensor(image)

        # MASK
        mask = Image.open(mask_path).resize((224,224))
        mask = np.array(mask)
        mask = (mask == 1).astype(np.float32)
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

        # LABEL
        breed = "_".join(name.split("_")[:-1])
        label = torch.tensor(self.breed_to_idx[breed])

        return {
            "image": image,
            "label": label,
            "bbox": bbox,
            "mask": mask
        }