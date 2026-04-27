## Regression Test

```python
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.getcwd(), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
```

```python
from src.datasets.oxford_pets import get_dataloader
from src.models.backbone import CNNModel, get_pretrained_model
from src.models.regressor import RectRegressor, RectRegressorNorm, RectIOU
from src.trainer import fit
from src.utils import show_image_rect
```

```python
iou = RectIOU()
os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"

train_loader = get_dataloader(data_dir, "train", task="regression")
batch = next(iter(train_loader))
images = batch["image"]
labels = batch["label"]
coords = batch["rect"]
coords_norm = batch["rect_norm"]

print(f"Images: {images.shape}, {images.dtype}")
print(f"Labels: {labels.shape}, {labels.dtype}")
print(f"Coords: {coords.shape}, {coords.dtype}")
print(f"Coords: {coords_norm.shape}, {coords_norm.dtype}")
for coord, coord_norm in zip(coords, coords_norm):
    print(coord, coord_norm)
```

```python
model1 = RectRegressor(model=get_pretrained_model("mobilenet_v3_small", output_dim=4))
_ = fit(model1, train_loader, max_epochs=10)
```

```python
model1_norm = RectRegressorNorm(model=get_pretrained_model("mobilenet_v3_small", output_dim=4))
_ = fit(model1_norm, train_loader, max_epochs=10)
```

```python
preds1 = model1.predict(images)
idx = 3
print(iou(coords[idx], preds1[idx]).item())
show_image_rect(images[idx], coords[idx])
show_image_rect(images[idx], preds1[idx])
```

```python
preds1_norm = model1_norm.predict(images)
idx = 3
print(iou(coords[idx], preds1_norm[idx]*224).item())
show_image_rect(images[idx], coords[idx])
show_image_rect(images[idx], preds1_norm[idx]*224)
```
