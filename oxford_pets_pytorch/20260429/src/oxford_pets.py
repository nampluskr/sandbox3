import os
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors


def get_samples(data_dir, split="train", exclude_corrupt=True, task="classification"):
    if split == 'train':
        split_file = os.path.join(data_dir, 'annotations', 'trainval.txt')
    elif split == 'test':
        split_file = os.path.join(data_dir, 'annotations', 'test.txt')
    else:
        raise ValueError(f"split must be 'train' or 'test': {split}")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    images_png = [
        "Egyptian_Mau_14", "Egyptian_Mau_139", "Egyptian_Mau_145", "Egyptian_Mau_156",
        "Egyptian_Mau_167", "Egyptian_Mau_177", "Egyptian_Mau_186", "Egyptian_Mau_191",
        "Abyssinian_5", "Abyssinian_34",
    ]
    images_corrupt = ["chihuahua_121", "beagle_116"]
    exclude_list = images_png + images_corrupt if exclude_corrupt else []

    samples = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            filename, label = parts[0], parts[1]
            if filename in exclude_list:
                continue

            image_path = os.path.join(data_dir, 'images', f'{filename}.jpg')
            if not os.path.exists(image_path):
                continue

            sample = {"image_path": image_path, "label": int(label) - 1}

            if task == "classification":
                samples.append(sample)
            elif task == "segmentation":
                mask_path = os.path.join(data_dir, 'annotations', 'trimaps', f'{filename}.png')
                if os.path.exists(mask_path):
                    sample["mask_path"] = mask_path
                    samples.append(sample)
            elif task in ["detection", "regression_rect", "regression_poly"]:
                xml_path = os.path.join(data_dir, 'annotations', 'xmls', f'{filename}.xml')
                if os.path.exists(xml_path):
                    sample["xml_path"] = xml_path
                    samples.append(sample)
            else:
                raise ValueError(f"Unsupported task: {task}")
    return samples


def get_transforms(split="train", img_size=512):
    transforms = [
        T.Resize((img_size, img_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    if split == "train":
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(2, 2)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


class OxfordPetsClassification(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.samples = get_samples(data_dir, split, task="classification")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = tv_tensors.Image(image)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToDtype(torch.float32, scale=True)(image)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return {
            "image": image,
            "label": label,
        }


def parse_xml_rect(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.find("xmin").text) - 1
            ymin = float(bndbox.find("ymin").text) - 1
            xmax = float(bndbox.find("xmax").text) - 1
            ymax = float(bndbox.find("ymax").text) - 1

            if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
                continue

            return [xmin, ymin, xmax, ymax]

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return []


class OxfordPetsRegressionRect(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.samples = get_samples(data_dir, split, task="regression_rect")
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        w, h = image.size

        bbox = parse_xml_rect(sample["xml_path"])
        coords = torch.zeros((1, 4), dtype=torch.float32)
        if len(bbox) == 4:
            coords[0] = torch.tensor(bbox, dtype=torch.float32)

        label = torch.tensor(sample["label"], dtype=torch.long)
        image = tv_tensors.Image(image)

        if self.transform:
            boxes = tv_tensors.BoundingBoxes(coords, format="XYXY", canvas_size=(h, w))
            image, coords = self.transform(image, boxes)

        coords = coords.view(-1)
        img_h, img_w = image.shape[-2:]
        coords_norm = coords / torch.tensor([img_w, img_h] * 2, dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "coord": coords,
            "coord_norm": coords_norm,
        }


def parse_xml_poly(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.find("xmin").text) - 1
            ymin = float(bndbox.find("ymin").text) - 1
            xmax = float(bndbox.find("xmax").text) - 1
            ymax = float(bndbox.find("ymax").text) - 1

            if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
                continue

            x1, y1 = xmin, ymin  # 좌상단
            x2, y2 = xmax, ymin  # 우상단
            x3, y3 = xmax, ymax  # 우하단
            x4, y4 = xmin, ymax  # 좌하단
            return [x1, y1, x2, y2, x4, y4, x3, y3]

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return []


def sort_clockwise(coords):
    squeeze = coords.dim() == 1
    if squeeze:
        coords = coords.unsqueeze(0)

    batch_size = coords.shape[0]
    pts = coords.view(batch_size, 4, 2)
    cx = pts[:, :, 0].mean(dim=1, keepdim=True)
    cy = pts[:, :, 1].mean(dim=1, keepdim=True)
    angles = torch.atan2(pts[:, :, 1] - cy, pts[:, :, 0] - cx)                                              # (B, 4)
    idx = angles.argsort(dim=1)
    sorted_pts = torch.stack([pts[i][idx[i]] for i in range(batch_size)], dim=0)                                              # (B, 4, 2)

    result = []
    for batch_idx in range(batch_size):
        pt = sorted_pts[batch_idx]
        start = (pt[:, 0] + pt[:, 1]).argmin().item()
        result.append(torch.roll(pt, -start, dims=0))

    return torch.stack(result).view(batch_size, 8).squeeze(0) if squeeze else \
           torch.stack(result).view(batch_size, 8)


class OxfordPetsRegressionPoly(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.samples = get_samples(data_dir, split, task="regression_poly")
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        w, h = image.size

        bbox = parse_xml_poly(sample["xml_path"])
        coords = torch.zeros((2, 4), dtype=torch.float32)

        if len(bbox) == 8:
            coords[0] = torch.tensor(bbox[:4], dtype=torch.float32)
            coords[1] = torch.tensor(bbox[4:], dtype=torch.float32)

        label = torch.tensor(sample["label"], dtype=torch.long)
        image = tv_tensors.Image(image)

        if self.transform:
            boxes = tv_tensors.BoundingBoxes(coords, format="XYXY", canvas_size=(h, w))
            image, coords = self.transform(image, boxes)

        coords = coords.view(-1)
        img_h, img_w = image.shape[-2:]
        coords_norm = coords / torch.tensor([img_w, img_h] * 4, dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "coord": sort_clockwise(coords),
            "coord_norm": sort_clockwise(coords_norm),
        }

    # def __getitem__(self, idx):s
    #     """torchvision.tv_tensors.KeyPoints는 TorchVision 0.17 이상에서만 지원"""
    #     sample = self.samples[idx]
    #     image = Image.open(sample["image_path"]).convert("RGB")
    #     w, h = image.size

    #     bbox = parse_xml_poly(sample["xml_path"])
    #     coords = torch.zeros((4, 2), dtype=torch.float32)

    #     if len(bbox) == 8:
    #         coords = torch.tensor(bbox, dtype=torch.float32).view(4, 2)

    #     label = torch.tensor(sample["label"], dtype=torch.long)
    #     image = tv_tensors.Image(image)

    #     if self.transform:
    #         kpts = tv_tensors.KeyPoints(coords, canvas_size=(h, w))
    #         image, coords = self.transform(image, kpts)

    #     coords = coords.view(-1)
    #     img_h, img_w = image.shape[-2:]
    #     coords_norm = coords / torch.tensor([img_w, img_h] * 4, dtype=torch.float32)

    #     return {
    #         "image": image,
    #         "label": label,
    #         "coord": kpts_flat,
    #         "coord_norm": kpts_norm
    #     }

def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        values = [b[key] for b in batch]

        if key in ["image", "mask"]:
            result[key] = torch.stack(values)
        elif key == "target":
            result[key] = values
        else:
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = torch.tensor(values)
    return result


def get_dataloader(data_dir, split="train", task="classification", img_size=512):
    transforms = get_transforms(split, img_size=img_size)

    if task == "classification":
        dataset = OxfordPetsClassification(data_dir, split, transform=transforms)
        batch_size = 16 if split == "train" else 8
    elif task == "regression_rect":
        dataset = OxfordPetsRegressionRect(data_dir, split, transform=transforms)
        batch_size = 16 if split == "train" else 16
    elif task == "regression_poly":
        dataset = OxfordPetsRegressionPoly(data_dir, split, transform=transforms)
        batch_size = 16 if split == "train" else 16
    # elif task == "detection":
    #     dataset = OxfordPetsDetection(data_dir, split, transform=transforms)
    #     batch_size = 4 if split == "train" else 2
    else:
        raise ValueError(f"Unsupported task: {task}")

    kwargs = {
        "batch_size": batch_size,
        "shuffle": split == "train",
        "drop_last": split == "train",
        "collate_fn": collate_fn,
        "pin_memory": True,
        "num_workers": 8,
        "persistent_workers": split == "train",
        "prefetch_factor": 2,
    }
    return DataLoader(dataset, **kwargs)
