# experiments/train.py

import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse

from src.config import load_config, save_config
from src.dataloader import get_split_dataloader
from src.backbone import build_model
from src.regressor import Regressor
from src.trainer import fit, fit_early_stop, evaluate, get_info
from src.utils import set_seed
from src.inference import save_weights, load_weights
from src.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--no_logging", action="store_true")
    return parser.parse_args()


def set_config(args):
    config_default = load_config(os.path.join(ROOT_DIR, "configs", "default.yaml"))
    config = {
        "dataset": args.dataset,
        "image_dir": config_default[args.dataset]["image_dir"],
        "csv_path": config_default[args.dataset]["csv_path"],
        "sampling": config_default[args.dataset]["sampling"],
        "train_image_dir": config_default[args.dataset]["train_image_dir"],
        "train_csv_path": config_default[args.dataset]["train_csv_path"],
        "test_image_dir": config_default[args.dataset]["test_image_dir"],
        "test_csv_path": config_default[args.dataset]["test_csv_path"],
        "backbone_dir": config_default["backbone_dir"],
        "backbone": args.backbone,
        "output_dim": 8,
        "batch_size": args.batch_size or 8,
        "image_size": args.image_size or 256,
        "max_epoch": args.max_epoch or 10,
        "seed": 42,
        "output_dir": os.path.join(ROOT_DIR, "outputs_test", args.dataset, args.backbone),
    }
    return config


def main():
    args = parse_args()
    config = set_config(args)

    experiment_name = f"{config['dataset']}-{config['backbone']}-img{config['image_size']}"
    log_path = None if args.no_logging else os.path.join(config["output_dir"], f"{experiment_name}.log")
    logger = get_logger(log_path)

    #################################################################
    ## Training
    #################################################################
    logger.info(" ")
    logger.info("*** Training:")

    set_seed(seed=config["seed"])
    train_loader = get_base_dataloader(
        image_dir=config["train_image_dir"],
        csv_path=config["train_csv_path"],
        transform=get_transform("train", image_size=config["image_size"]),
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    test_loader = get_base_dataloader(
        image_dir=config["test_image_dir"],
        csv_path=config["test_csv_path"],
        transform=get_transform("train", image_size=config["image_size"]),
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    model = build_model(pretrained=True, **config)
    trainer = Regressor(model)

    if args.early_stop:
        _, best_epoch = fit_early_stop(trainer, train_loader, valid_loader=test_loader, 
                                       monitor="loss", mode="min", patience=5, **config)
    else:
        _, best_epoch = fit(trainer, train_loader, valid_loader=test_loader, **config)

    config["experiment_name"] = experiment_name
    config["best_epoch"] = best_epoch
    config["weights_path"] = os.path.join(config["output_dir"], f"{experiment_name}-ep{best_epoch:03d}.pth")
    config["config_path"] = os.path.join(config["output_dir"], f"{experiment_name}-ep{best_epoch:03d}.yaml")

    save_weights(model, weights_path=config["weights_path"])
    save_config(config, config_path=config["config_path"])

    del model, trainer, train_loader

    #################################################################
    ## Evaluation
    #################################################################
    logger.info(" ")
    logger.info("*** Evaluation:")
    model_trained = build_model(pretrained=False, **config)
    load_weights(model_trained, weights_path=config["weights_path"])

    trainer = Regressor(model_trained)
    results = evaluate(trainer, test_loader)
    logger.info(f"> {get_info(results)}")

    del model_trained, trainer, test_loader

if __name__ == "__main__":
    main()
