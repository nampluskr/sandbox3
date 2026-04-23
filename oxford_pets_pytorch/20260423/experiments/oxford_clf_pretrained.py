import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.utils import set_seed
from src.datasets.oxford_pets import get_dataloader
from src.models.classifier import MulticlassClassifier
from src.models.backbone import CNNModel, get_pretrained_model
from src.trainer import fit


# def main():
#     set_seed(42)
#     data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"
#     train_loader = get_dataloader(data_dir, "train", task="classification")
#     test_loader = get_dataloader(data_dir, "test", task="classification")

#     model = MulticlassClassifier(
#         model=CNNModel(output_dim=37, in_channels=3),
#         num_classes=37,
#     )
#     fit(model, train_loader, max_epochs=10, valid_loader=test_loader)

def main():
    set_seed(42)
    data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"
    train_loader = get_dataloader(data_dir, "train", task="classification")
    test_loader = get_dataloader(data_dir, "test", task="classification")

    os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
    model = MulticlassClassifier(
        model=get_pretrained_model("mobilenet_v3_small", output_dim=37),
        num_classes=37,
    )
    fit(model, train_loader, max_epochs=10, valid_loader=test_loader)


if __name__ == "__main__":
    main()
