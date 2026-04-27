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
            elif task in ["detection", "regression"]:
                xml_path = os.path.join(data_dir, 'annotations', 'xmls', f'{filename}.xml')
                if os.path.exists(xml_path):
                    sample["xml_path"] = xml_path
                    samples.append(sample)
            else:
                raise ValueError(f"Unsupported task: {task}")
    return samples


def parse_xml(xml_path):
    boxes = []
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
            if xmin >= 0 and ymin >= 0 and xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return boxes


# def get_transforms(split="train", img_size=224):
#     transforms = [
#         T.ToImage(),
#         T.ToDtype(torch.float32, scale=True),
#         T.Resize((img_size, img_size)),
#     ]
#     if split == "train":
#         transforms.extend([
#             T.RandomHorizontalFlip(p=0.5),
#         ])
#     transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
#     return T.Compose(transforms)


def get_transforms(split="train", img_size=224):
    transforms = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((256, 256)),
    ]
    if split == "train":
        transforms.extend([
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
    else:
        transforms.append(T.CenterCrop(img_size))

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


class OxfordPetsRegression(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.samples = get_samples(data_dir, split, task="regression")
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        w, h = image.size

        bbox_list = parse_xml(sample["xml_path"])
        if len(bbox_list) == 0:
            box = [0.0, 0.0, 0.0, 0.0]
        else:
            box = bbox_list[0]

        coords = torch.tensor(box, dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        image = tv_tensors.Image(image)

        if self.transform:
            boxes = tv_tensors.BoundingBoxes(coords.unsqueeze(0), format="XYXY", canvas_size=(h, w))
            image, boxes = self.transform(image, boxes)
            coords = boxes.squeeze(0)

        img_h, img_w = image.shape[-2:]
        coords_norm = coords / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "rect": coords,
            "rect_norm": coords_norm,
        }


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


def get_dataloader(data_dir, split="train", task="classification"):
    transforms = get_transforms(split)

    if task == "classification":
        dataset = OxfordPetsClassification(data_dir, split, transform=transforms)
        batch_size = 16 if split == "train" else 8
    elif task == "regression":
        dataset = OxfordPetsRegression(data_dir, split, transform=transforms)
        batch_size = 16 if split == "train" else 16
    elif task == "detection":
        dataset = OxfordPetsDetection(data_dir, split, transform=transforms)
        batch_size = 4 if split == "train" else 2
    else:
        raise ValueError(f"Unsupported task: {task}")

    kwargs = {
        "batch_size": batch_size,
        "shuffle": split == "train",
        "drop_last": split == "train",
        "collate_fn": collate_fn,
        "pin_memory": True,
        "num_workers": 4,
        "persistent_workers": split == "train",
        "prefetch_factor": 2,
    }
    return DataLoader(dataset, **kwargs)
