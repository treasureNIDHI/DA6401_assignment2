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

        self.samples = []
        self.sample_to_label = {}
        with open(split_file) as f:
            # Split file format: image_name class_id species_id breed_id
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                sample_name = parts[0]
                # Official class ids are 1..37; convert to 0..36 for CE loss.
                class_id = int(parts[1]) - 1

                xml_path = os.path.join(self.xml_dir, sample_name + ".xml")
                if os.path.exists(xml_path):
                    self.samples.append(sample_name)
                    self.sample_to_label[sample_name] = class_id

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Handle rare corrupted files gracefully by trying a nearby sample.
        for offset in range(len(self.samples)):
            cur_idx = (idx + offset) % len(self.samples)
            name = self.samples[cur_idx]

            img_path = os.path.join(self.images_dir, name + ".jpg")
            mask_path = os.path.join(self.masks_dir, name + ".png")
            xml_path = os.path.join(self.xml_dir, name + ".xml")

            try:
                # IMAGE
                image = Image.open(img_path).convert("RGB")
                original_width, original_height = image.size
                image = image.resize((224,224))
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = self.to_tensor(image)

                # MASK
                mask = Image.open(mask_path).resize((224,224), resample=Image.NEAREST)
                mask = np.array(mask, dtype=np.int64) - 1
                mask = torch.tensor(mask, dtype=torch.long)

                # BBOX
                tree = ET.parse(xml_path)
                root = tree.getroot()

                bbox = root.find("object").find("bndbox")

                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                scale_x = 224.0 / float(original_width)
                scale_y = 224.0 / float(original_height)

                xmin *= scale_x
                xmax *= scale_x
                ymin *= scale_y
                ymax *= scale_y

                # convert to center format
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                bbox = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

                # LABEL
                label = torch.tensor(self.sample_to_label[name], dtype=torch.long)

                return {
                    "image": image,
                    "label": label,
                    "bbox": bbox,
                    "mask": mask
                }
            except (OSError, ET.ParseError, ValueError):
                continue

        raise RuntimeError("No valid sample could be loaded from dataset.")