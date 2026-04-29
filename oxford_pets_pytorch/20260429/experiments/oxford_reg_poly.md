```python
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.getcwd(), '..') )
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from src.oxford_pets import get_dataloader
from src.backbone import CNNModel, get_pretrained_model
from src.regressor import Regressor, RectIOU, PolyIOU
from src.trainer import fit
from src.utils import set_seed, show_image_rect, show_image_poly

set_seed(42)
rect_iou = RectIOU()
poly_iou = PolyIOU()
os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"

train_loader = get_dataloader(data_dir, "train", task="regression_poly", img_size=512)
batch = next(iter(train_loader))
images = batch["image"]
labels = batch["label"]
coords = batch["coord"]
coords_norm = batch["coord_norm"]

print(f"Images: {images.shape}, {images.dtype}")
print(f"Labels: {labels.shape}, {labels.dtype}")
print(f"Coords: {coords.shape}, {coords.dtype}")
print(f"Coords: {coords_norm.shape}, {coords_norm.dtype}")

# for coord in coords:
#     print(coord)

# for coord_norm in coords_norm:
#     print(coord_norm)

# idx = 1
# show_image_poly(images[idx], coords[idx])

model = get_pretrained_model("mobilenet_v3_large", output_dim=8)
reg = Regressor(model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    use_sigmoid=True, 
    iou=PolyIOU()
)
_ = fit(reg, train_loader, max_epochs=10)

preds = reg.predict(images)

idx = 2
h, w = images[idx].shape[-2:]
image = images[idx]
coord = coords[idx]
pred = preds[idx] * torch.tensor([h, w] * 4)

print(f"({h}, {w}), {poly_iou(coord, pred).item():.2f}")
show_image_poly(image, coord)
show_image_poly(image, pred)
```
