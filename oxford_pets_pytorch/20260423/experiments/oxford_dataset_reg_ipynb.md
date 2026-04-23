```python
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.getcwd(), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
```

```python
from src.datasets.oxford_pets import get_dataloader
data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"

print(f"\n>> Regression:")
det_dataloader = get_dataloader(data_dir, "train", task="regression")
det_batch = next(iter(det_dataloader))
images = det_batch["image"]
labels = det_batch["label"]
coords = det_batch["coord"]

print(f"Images: {images.shape}, {images.dtype}")
print(f"Labels: {labels.shape}, {labels.dtype}")
print(f"Coords: {coords.shape}, {coords.dtype}")
for coord in coords:
    print(coord)
```

```python
from src.utils import show_image_rectangle, show_image_polygon

i = 1
show_image_rectangle(images[i], coords[i])
show_image_polygon(images[i], coords[i])
```

```python
from src.utils import set_seed
from src.datasets.oxford_pets import get_dataloader
from src.models.regressor import Regressor
from src.models.backbone import CNNModel, get_pretrained_model
from src.trainer import fit, train, evaluate

set_seed(42)
data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"
train_loader = get_dataloader(data_dir, "train", task="regression")
test_loader = get_dataloader(data_dir, "test", task="regression")

os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
model = Regressor(model=get_pretrained_model("wide_resnet50_2", output_dim=8))
history = fit(model, train_loader, max_epochs=10)
```

```python
print(f"{images.shape}")
print(f"{coords.shape}")

preds = model.predict(images).view(-1, 2, 4)
print(f"{preds.shape}")
```

```python
idx = 7
show_image_polygon(images[idx], coords[idx])
show_image_polygon(images[idx], preds[idx])
```



