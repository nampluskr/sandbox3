```python
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.getcwd(), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.datasets.oxford_pets import get_dataloader
from src.models.backbone import CNNModel, get_pretrained_model
from src.models.classifier import MulticlassClassifier
from src.trainer import fit
from src.utils import set_seed

set_seed(42)
os.environ["BACKBONE_DIR"] = "D:\\Non_Documents\\backbones"
data_dir = "E:\\datasets\\oxford_pets"

train_loader = get_dataloader(data_dir, "train", task="classification")
test_loader = get_dataloader(data_dir, "test", task="classification")
batch = next(iter(train_loader))
images = batch["image"]
labels = batch["label"]

print(f"Images: {images.shape}, {images.dtype}")
print(f"Labels: {labels.shape}, {labels.dtype}")

model = MulticlassClassifier(model=CNNModel(output_dim=37), num_classes=37)
_ = fit(model, train_loader, max_epochs=10, valid_loader=test_loader)

model = MulticlassClassifier(
        model=get_pretrained_model("mobilenet_v3_small", output_dim=37),
        num_classes=37,
)
_ = fit(model, train_loader, max_epochs=10, valid_loader=test_loader)
```
