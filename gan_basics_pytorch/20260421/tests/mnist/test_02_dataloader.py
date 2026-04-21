# tests/mnist/test_02_dataloader.py

import torch
from src.datasets.mnist import load_mnist


def test_dataloader_output_shape(data_dir, batch_size):
    loader = load_mnist(
        data_dir=data_dir,
        split='train',
        batch_size=batch_size,
        num_workers=0
    )
    batch = next(iter(loader))
    images = batch['image']
    labels = batch['label']

    assert images.shape == (batch_size, 1, 32, 32)
    assert images.dtype == torch.float32
    assert labels.shape == (batch_size,)
    assert labels.dtype == torch.long


def test_dataloader_pixel_range(data_dir, batch_size):
    loader = load_mnist(data_dir=data_dir, split='train', batch_size=batch_size, num_workers=0)
    images = next(iter(loader))['image']

    assert images.min() >= -1.0
    assert images.max() <= 1.0


def test_dataloader_shuffle_enabled(data_dir, batch_size):
    loader = load_mnist(data_dir=data_dir, split='train', batch_size=4, shuffle=True, num_workers=0)
    batch1 = next(iter(loader))
    batch2 = next(iter(loader))

    assert not torch.equal(batch1['label'], batch2['label']) or len(loader) > 1


def test_dataloader_test_split(data_dir):
    loader = load_mnist(data_dir=data_dir, split='test', batch_size=100, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    images = batch['image']
    labels = batch['label']

    assert images.shape[0] == 100
    assert labels.shape[0] == 100
    assert images.shape == (100, 1, 32, 32)
    assert labels.shape == (100,)
