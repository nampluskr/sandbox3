# src/models/weights.py

import os
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def save_weights(model, save_path, optimizer=None, epoch=None, metrics=None):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    checkpoint = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, save_path)


def load_weights(model, load_path, optimizer=None, strict=True):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Weights file not found: {load_path}")

    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
