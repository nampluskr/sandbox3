# tests/mnist/test_01_dataset.py

import torch
from src.datasets.mnist import MNISTDataset


def test_mnist_dataset_length(data_dir):
    train_dataset = MNISTDataset(data_dir=data_dir, split='train')
    test_dataset = MNISTDataset(data_dir=data_dir, split='test')
    assert len(train_dataset) == 60000
    assert len(test_dataset) == 10000


def test_mnist_dataset_item_structure(data_dir, mnist_transform):
    dataset = MNISTDataset(data_dir=data_dir, split='train', transform=mnist_transform)
    sample = dataset[0]
    assert 'image' in sample
    assert 'label' in sample


def test_mnist_dataset_image_shape(data_dir, mnist_transform):
    dataset = MNISTDataset(data_dir=data_dir, split='train', transform=mnist_transform)
    image = dataset[0]['image']
    assert image.shape == (1, 32, 32)
    assert image.dtype == torch.float32


def test_mnist_dataset_image_range(data_dir, mnist_transform):
    dataset = MNISTDataset(data_dir=data_dir, split='train', transform=mnist_transform)
    image = dataset[0]['image']
    assert image.min() >= -1.0
    assert image.max() <= 1.0


def test_mnist_dataset_label_type(data_dir, mnist_transform):
    dataset = MNISTDataset(data_dir=data_dir, split='train', transform=mnist_transform)
    label = dataset[0]['label']
    assert isinstance(label, torch.Tensor)
    assert label.shape == ()
    assert label.dtype == torch.long
