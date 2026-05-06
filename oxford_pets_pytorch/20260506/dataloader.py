# src/dataloader.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors


def sort_clockwise(coords):
    pts = coords.view(4, 2)
    center = pts.mean(dim=0)
    angles = torch.atan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_pts = pts[angles.argsort()]
    start_idx = (sorted_pts[:, 0] + sorted_pts[:, 1]).argmin()
    return torch.roll(sorted_pts, -start_idx.item(), dims=0).view(-1)


class BaseDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        self.samples = []
        for _, row in df.iterrows():
            self.samples.append({
                "image_path": os.path.join(image_dir, row["image_name"]),
                "bbox": [
                    row["x1"], row["y1"],  # tl (top left)
                    row["x2"], row["y2"],  # tr (top right)
                    row["x3"], row["y3"],  # br (bottom right)
                    row["x4"], row["y4"]   # bl (bottom left)
                ]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image_path"])
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        image = tv_tensors.Image(image)

        bbox = sample["bbox"]
        coords = torch.zeros((2, 4), dtype=torch.float32)

        if len(bbox) == 8:
            coords[0] = torch.tensor(bbox[:4], dtype=torch.float32)
            coords[1] = torch.tensor(bbox[4:], dtype=torch.float32)

        if self.transform:
            boxes = tv_tensors.BoundingBoxes(coords, format="XYXY", canvas_size=(h, w))
            image, coords = self.transform(image, boxes)

        coords = sort_clockwise(coords.view(-1))
        img_h, img_w = image.shape[-2:]
        coords_norm = coords / torch.tensor([img_w, img_h] * 4, dtype=torch.float32)
        return {
            "image": image,
            "coord": coords,
            "coord_norm": coords_norm,
        }


class SplitDataset(BaseDataset):
    def __init__(self, split, image_dir, csv_path, transform=None,
                 sampling=1.0, test_size=0.2, seed=42):
        if split not in ["train", "test"]:
            raise ValueError(f"split must be 'train' or 'test': {split}")

        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        coord_cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        df = df[
            df[coord_cols].notna().all(axis=1) &
            (df[coord_cols] > 0).all(axis=1) &
            (df[coord_cols] < 1e6).all(axis=1) &
            (~df[coord_cols].isin([float('inf'), float('-inf')]).any(axis=1))
        ].reset_index(drop=True)
        df = df.sample(frac=sampling, random_state=seed).reset_index(drop=True)

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
        df = train_df if split == "train" else test_df
        df = df.reset_index(drop=True)

        self.samples = []
        for _, row in df.iterrows():
            self.samples.append({
                "image_path": os.path.join(image_dir, row["image_name"]),
                "bbox": [
                    row["x1"], row["y1"],  # tl (top left)
                    row["x2"], row["y2"],  # tr (top right)
                    row["x3"], row["y3"],  # br (bottom right)
                    row["x4"], row["y4"]   # bl (bottom left)
                ]
            })


def get_transform(split, image_size=256):
    if split not in ["train", "test"]:
        raise ValueError(f"split must be 'train' or 'test': {split}")

    transforms = [
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    if split == "train":
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(1, 2)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.5),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0)),
            T.RandomRotation(degrees=(90, 90)),
            T.RandomRotation(degrees=(-90, -90)),
        ])
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        values = [b[key] for b in batch]
        if key in ["image"]:
            result[key] = torch.stack(values)
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = torch.tensor(values)
    return result


def get_base_dataloader(image_dir, csv_path, transform,
    batch_size=16, shuffle=True, drop_last=True, **kwargs):
    dataset = BaseDataset(image_dir, csv_path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8,
        # persistent_workers=(split == "train"),
        # prefetch_factor=2,
    )


def get_split_dataloader(split, image_dir, csv_path, sampling=1.0, batch_size=16, image_size=256,
                   test_size=0.2, seed=42, **kwargs):
    if split not in ["train", "test"]:
        raise ValueError(f"split must be 'train' or 'test': {split}")

    dataset = SplitDataset(
        split=split,
        image_dir=image_dir,
        csv_path=csv_path,
        transform=get_transform(split=split, image_size=image_size),
        sampling=sampling,
        test_size=test_size,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8,
        # persistent_workers=(split == "train"),
        # prefetch_factor=2,
    )


def get_combined_dataloader(dataset_configs, split, batch_size=16, image_size=256, **kwargs):
    """
        config = load_config(os.path.join("configs", "default.yaml"))
        dataset_configs = [config["oxford"], config["midv2020"], config["smartdoc"]]
    """
    datasets = []
    for config in dataset_configs:
        dataset = SplitDataset(
            split=split,
            image_dir=config["image_dir"],
            csv_path=config["csv_path"],
            transform=get_transform(split=split, image_size=image_size),
            sampling=config.get("sampling", 1.0),
        )
        datasets.append(dataset)

    combined = ConcatDataset(datasets)

    if split == "train":
        dataset_weights = [config.get("weight", 1.0) for config in dataset_configs]
        sample_weights = []
        for dataset, w in zip(datasets, dataset_weights):
            sample_weights.extend([w] * len(dataset))

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(combined), replacement=True)
        return DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=8,
        )
    else:
        return DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=8,
        )


