import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

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


def to_numpy(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img


def show_image_rectangle(image, coords):
    fig, ax = plt.subplots()
    ax.imshow(to_numpy(image))
    x1, y1, x2, y2 = coords[0].cpu().numpy()
    x3, y3, x4, y4 = coords[1].cpu().numpy()
    ax.add_patch(Rectangle((x1, y1), x3 - x1, y3 - y1,
        linewidth=2, edgecolor='red', facecolor='none'))
    plt.axis('off')
    fig.tight_layout()
    plt.show()


def show_image_polygon(image, coords):
    fig, ax = plt.subplots()
    ax.imshow(to_numpy(image))
    x1, y1, x2, y2 = coords[0].cpu().numpy()
    x3, y3, x4, y4 = coords[1].cpu().numpy()
    ax.add_patch(Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        closed=True, linewidth=2, edgecolor='red', facecolor='none'))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
