# src/trainer.py

import os
import sys
from tqdm import tqdm


def get_info(results):
    return ", ".join([f"{name}:{value:.3f}" for name, value in results.items()])


def train(model, dataloader):
    model.train()
    metrics = {}
    total_size = 0

    with tqdm(dataloader, desc="Train", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            train_results = model.train_step(batch)
            total_size += train_results["batch_size"]

            for name, value in train_results.items():
                if name != "batch_size":
                    metrics.setdefault(name, 0.0)
                    metrics[name] += float(value) * train_results["batch_size"]

            info = {name: f"{value / total_size:.3f}" for name, value in metrics.items()}
            progress_bar.set_postfix(info)

    return {name: value / total_size for name, value in metrics.items()}


def evaluate(model, dataloader):
    model.eval()
    metrics = {}
    total_size = 0

    with tqdm(dataloader, desc="Evaluate", file=sys.stdout, leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            test_results = model.eval_step(batch)
            total_size += test_results["batch_size"]

            for name, value in test_results.items():
                if name != "batch_size":
                    metrics.setdefault(name, 0.0)
                    metrics[name] += float(value) * test_results["batch_size"]

            info = {name: f"{value / total_size:.3f}" for name, value in metrics.items()}
            progress_bar.set_postfix(info)

    return {name: value / total_size for name, value in metrics.items()}


def fit(model, train_loader, max_epochs, valid_loader=None):
    history = {"train": {}, "valid": {}}
    for epoch in range(1, max_epochs + 1):
        train_results = train(model, train_loader)
        epoch_info = f"[{epoch:3d}/{max_epochs}]"

        for name, value in train_results.items():
            history["train"].setdefault(name, [])
            history["train"][name].append(value)

        if valid_loader is not None:
            valid_results = evaluate(model, valid_loader)

            for name, value in valid_results.items():
                history["valid"].setdefault(name, [])
                history["valid"][name].append(value)
            print(f"{epoch_info} {get_info(train_results)} | (val) {get_info(valid_results)}")
        else:
            print(f"{epoch_info} {get_info(train_results)}")

    return history
