import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.utils import set_seed
from src.datasets.oxford_pets import get_dataloader
from src.models.regressor import Regressor
from src.models.backbone import CNNModel, get_pretrained_model
from src.trainer import fit, train, evaluate


# def main():
#     set_seed(42)
#     data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"
#     train_loader = get_dataloader(data_dir, "train", task="regression")
#     test_loader = get_dataloader(data_dir, "test", task="regression")

#     model = Regressor(model=CNNModel(output_dim=8, in_channels=3))
#     fit(model, train_loader, max_epochs=10)


def main():
    set_seed(42)
    data_dir = "/home/namu/myspace/NAMU/datasets/oxford_pets"
    train_loader = get_dataloader(data_dir, "train", task="regression")
    test_loader = get_dataloader(data_dir, "test", task="regression")

    os.environ["BACKBONE_DIR"] = "/home/namu/myspace/NAMU/backbones"
    model = Regressor(model=get_pretrained_model("resnet18", output_dim=8))
    fit(model, train_loader, max_epochs=10)


if __name__ == "__main__":
    main()
