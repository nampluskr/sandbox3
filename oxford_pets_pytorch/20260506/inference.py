# src/inference.py

import os
import torch
from PIL import Image
import logging

logger = logging.getLogger("train")

def save_weights(model, weights_path):
    save_dir = os.path.dirname(weights_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), weights_path)
    logger.info(f"> Model weights saved: {os.path.basename(weights_path)}")


def load_weights(model, weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"> Model weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"> Model weights loaded: {os.path.basename(weights_path)}")


@torch.no_grad()
def predict(model, image_path, image_size=256, device=None, transform=None):
    from src.dataloader import get_transform, sort_clockwise
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    transform = transform or get_transform(split="test", image_size=image_size)

    image_tensor = transform(image).unsqueeze(0).to(device)
    logits = model(image_tensor)
    preds = torch.sigmoid(logits).cpu()
    preds = preds * torch.tensor([w, h] * 4)
    return preds.squeeze(0).tolist()
