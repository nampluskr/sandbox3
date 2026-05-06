import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.getcwd(), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from src.config import load_config, save_config
from src.dataloader import get_transform, get_base_dataloader
from src.backbone import build_model
from src.regressor import Regressor
from src.trainer import fit, fit_early_stop, evaluate, get_info
from src.utils import set_seed
from src.inference import save_weights, load_weights, predict
from src.logger import get_logger
from src.utils import show_image_poly

# config_path = "/home/namu/myspace/NAMU/clones/polygon_regression/outputs/combined/resnet34/combined-resnet34-img256-ep023.yaml"
config_path = "/home/namu/myspace/NAMU/clones/polygon_regression/outputs/combined/resnet34/combined-resnet34-img512-ep022.yaml"
config = load_config(config_path)

data_dir = "/home/namu/myspace/NAMU/clones/polygon_regression/data"
config["train_image_dir"] = os.path.join(data_dir, "train_images")
config["train_csv_path"] = os.path.join(data_dir, "anotations_data_train.csv")
config["test_image_dir"] = os.path.join(data_dir, "test_images")
config["test_csv_path"] = os.path.join(data_dir, "anotations_data_test.csv")
config["batch_size"] = 4

set_seed(seed=config["seed"])

train_transform = get_transform("train", image_size=config["image_size"])
train_loader = get_base_dataloader(
    image_dir=config["train_image_dir"],
    csv_path=config["train_csv_path"],
    transform=train_transform,
    batch_size=config["batch_size"],
    shuffle=True,
    drop_last=False,
)

test_transform = get_transform("test", image_size=config["image_size"])
test_loader = get_base_dataloader(
    image_dir=config["test_image_dir"],
    csv_path=config["test_csv_path"],
    transform=test_transform,
    batch_size=1,
    shuffle=False,
    drop_last=False,
)

model = build_model(
    backbone=config["backbone"], 
    backbone_dir=config["backbone_dir"], 
    output_dim=8, 
    pretrained=False
)
load_weights(model, weights_path=config["weights_path"])
logger = get_logger()

# model = build_model(
#     backbone=config["backbone"], 
#     backbone_dir=config["backbone_dir"], 
#     output_dim=8, 
#     pretrained=True
# )
# # load_weights(model, weights_path=config["weights_path"])
# logger = get_logger()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = Regressor(model, optimizer=optimizer)
_, _ = fit(trainer, train_loader, max_epoch=10, valid_loader=test_loader, logger=logger)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
trainer = Regressor(model, optimizer=optimizer)
_, _ = fit(trainer, train_loader, max_epoch=10, valid_loader=test_loader, logger=logger)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
trainer = Regressor(model, optimizer=optimizer)
_, _ = fit(trainer, train_loader, max_epoch=10, valid_loader=test_loader, logger=logger)

for sample in test_loader.dataset.samples:
    image_path = sample["image_path"]
    target = sample["bbox"]
    pred = predict(model, image_path, image_size=config["image_size"], transform=test_transform)
    show_image_poly(image_path, target, pred=pred)
