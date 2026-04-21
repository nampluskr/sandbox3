# src/utils.py

import os
import random
import numpy as np
from matplotlib import pyplot as plt
import torch


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def make_sample_path(output_dir, path, epoch):
    filename = get_filename(path)
    suffix = f"ep{epoch:03d}.png"
    sample_path = os.path.join(output_dir, filename, f"{filename}_{suffix}")
    return sample_path


def create_noises(num_samples, latent_dim):
    return torch.randn(num_samples, latent_dim)


@torch.no_grad()
def create_images(generator, noises):
    generator.eval()
    device = next(generator.parameters()).device
    noises = noises.to(device)
    images = generator(noises)
    images = (images + 1) / 2.0
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    return images.squeeze(-1)


def plot_images(*images, ncols=5, xunit=1, yunit=1, cmap='gray',
                titles=[], vmin=None, vmax=None, save_path=None):
    num_images = len(images)
    ncols = min(ncols, num_images)
    nrows = (num_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * xunit, nrows * yunit))
    axes = np.array(axes).reshape(-1) if num_images > 1 else [axes]

    for idx, img in enumerate(images):
        axes[idx].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[idx].axis('off')
        if len(titles) > idx:
            axes[idx].set_title(titles[idx])

    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f">> {os.path.basename(save_path)} saved.\n")
    else:
        plt.show()
