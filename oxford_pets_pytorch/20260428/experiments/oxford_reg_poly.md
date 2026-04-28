```python
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.getcwd(), '..') )
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.datasets.oxford_pets import get_dataloader
from src.models.backbone import CNNModel, get_pretrained_model
from src.models.regressor import Regressor, RectIOU, PolyIOU
from src.trainer import fit
from src.utils import set_seed, show_image_rect, show_image_poly

set_seed(42)
rect_iou = RectIOU()
poly_iou = PolyIOU()
os.environ["BACKBONE_DIR"] = "D:\\Non_Documents\\backbones"
data_dir = "E:\\datasets\\oxford_pets"

train_loader = get_dataloader(data_dir, "train", task="regression_poly")
batch = next(iter(train_loader))
images = batch["image"]
labels = batch["label"]
coords = batch["coord"]
coords_norm = batch["coord_norm"]

print(f"Images: {images.shape}, {images.dtype}")
print(f"Labels: {labels.shape}, {labels.dtype}")
print(f"Coords: {coords.shape}, {coords.dtype}")
print(f"Coords: {coords_norm.shape}, {coords_norm.dtype}")

for coord in coords:
    print(coord)

for coord_norm in coords_norm:
    print(coord_norm)

idx = 1
show_image_poly(images[idx], coords[idx])

model1 = Regressor(model=get_pretrained_model("mobilenet_v3_small", output_dim=8), use_sigmoid=False)
_ = fit(model1, train_loader, max_epochs=10)

model1_norm = Regressor(model=get_pretrained_model("mobilenet_v3_small", output_dim=8), use_sigmoid=True)
_ = fit(model1_norm, train_loader, max_epochs=10)

preds1 = model1.predict(images)
idx = 1
print(rect_iou(coords[idx], preds1[idx]).item())
show_image_poly(images[idx], preds1[idx])

preds1_norm = model1_norm.predict(images)
idx = 1
print(rect_iou(coords[idx], preds1_norm[idx]*224).item())
show_image_poly(images[idx], preds1_norm[idx]*224)

model2 = Regressor(model=get_pretrained_model("mobilenet_v3_small", output_dim=8), use_sigmoid=False, iou=PolyIOU())
_ = fit(model2, train_loader, max_epochs=10)

model2_norm = Regressor(model=get_pretrained_model("mobilenet_v3_small", output_dim=8), use_sigmoid=True, iou=PolyIOU())
_ = fit(model2_norm, train_loader, max_epochs=10)

preds2 = model2.predict(images)
idx = 1
print(poly_iou(coords[idx], preds2[idx]).item())
show_image_poly(images[idx], preds2[idx])

preds2_norm = model2_norm.predict(images)
idx = 1
print(poly_iou(coords[idx], preds2_norm[idx]*224).item())
show_image_rect(images[idx], coords[idx])
show_image_rect(images[idx], preds2_norm[idx]*224)
```
