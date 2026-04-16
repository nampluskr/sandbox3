# sandbox3

- `[title]_[backend]_[dataset]_[task]_[model]`

```bash
mnist-numpy-mlp/ # mnist-cupy-cnn
├── configs/
├── src/
│   ├── common/
│   ├── modules/
│   └── trainers/
├── outputs/
└── experiments/
    ├── 01_multiclass/
    ├── 02_binary/
    ├── 03_regression/
    ├── 04_autoencoder/
    ├── 05_vae/
    └── 06_gan/
```

```bash
gan-basic-pytorch/  # basic / intetmediate / advanced
├── configs/
├── src/
│   ├── common/
│   ├── modules/
│   └── trainers/
├── outputs/
└── experiments/
    ├── 01_mnist/
    │   ├── 01_mnist_gan.py
    │   ├── 02_mnist_wgan.py
    │   ├── 03_mnist_cgan.py
    │   └── 04_mnist_acgan.py
    ├── 02_cifar10/
    └── 03_celeba/
```

```bash
anomaly-detection-pytorch/
├── configs/
├── src/
│   ├── common/
│   ├── models/
│   └── trainers/
├── outputs/
└── experiments/
    ├── 01_mvtec/
    │   ├── 01_mvtec_category_stfpm.py
    │   ├── 02_mvtec_category_dinomaly.py
    ├── 02_visa/
    └── 03_btad/
```
