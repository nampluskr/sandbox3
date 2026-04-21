# src/training/trainer.py

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

    results = {name: value / total_size for name, value in metrics.items()}
    return results
