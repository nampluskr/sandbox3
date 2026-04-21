# tests/mnist/conftest.py

import pytest
from src.datasets.mnist import get_transforms
from src.utils.config import load_config


@pytest.fixture(scope="session")
def data_dir():
    config = load_config('configs/mnist.yaml')
    return config['data']['data_dir']


@pytest.fixture(scope="session")
def img_size():
    return 32


@pytest.fixture(scope="session")
def mnist_transform(img_size):
    return get_transforms(img_size=img_size, split='train')


@pytest.fixture(scope="session")
def batch_size():
    config = load_config('configs/mnist.yaml')
    return config['data']['batch_size']
