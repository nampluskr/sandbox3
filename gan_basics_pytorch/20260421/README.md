# GAN Basics with Pytorch

### Structure

```
gan-basics-pytorch/
│
├── configs/
│   ├── mnist.yaml
│   └── gan.yaml
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   ├── gan.py
│   │   └── weights.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── mnist.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   │
│   ├── __init__.py
│   ├── config.py
│   └── utils.py
│
├── experiments/
│   └── mnist/
│       └── 01_mnist_gan.py
│
├── outputs/
│   └── mnist/
│       └── 01_mnist_gan/
│           ├── 01_mnist_gan_ep005.png
│           ├── 01_mnist_gan_ep010.png
│           ├── 01_mnist_gan_ep015.png
│           └── 01_mnist_gan_ep020.png
│
├── tests/
│   ├── conftest.py
│   └── mnist/
│       ├── conftest.py
│       ├── test_01_dataset.py
│       ├── test_02_dataloader.py
│       └── test_03_gan_model.py
│
├── requirements.txt
└── README.md
```
