# src/datasets/mnist.py

import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T


def get_class_names(mnist_type="mnist"):
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def load_images(data_dir, split="train"):
    filename = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28).copy()


def load_labels(data_dir, split="train"):
    filename = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def get_transforms(img_size=32, split="train"):
    return T.Compose([
        T.ToImage(),
        T.Pad((2, 2)),
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])


class MNISTDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.images = load_images(data_dir, split)
        self.labels = load_labels(data_dir, split)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        label = torch.tensor(label).long()
        return {"image": image, "label": label}


def load_mnist(data_dir, split="train", img_size=32, batch_size=64, shuffle=None, num_workers=4):
    transform = get_transforms(img_size, split)
    dataset = MNISTDataset(data_dir, split, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle or (split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )
    return loader
