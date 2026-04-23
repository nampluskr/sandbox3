import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


class Regressor(nn.Module):
    def __init__(self, model, optimizer=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)
        self.mse_metric = MeanSquaredError().to(self.device)
        self.mae_metric = MeanAbsoluteError().to(self.device)

    def train_step(self, batch):
        self.train()
        images = batch["image"].to(self.device)
        coords = batch["coord"].to(self.device)
        if coords.dim() == 3:
            coords = coords.view(coords.size(0), -1)

        preds = self.model(images)
        loss = self.loss_fn(preds, coords)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            mse = self.mse_metric(preds, coords)
            mae = self.mae_metric(preds, coords)
        return {
            "loss": loss.item(), 
            "mse": mse.item(), 
            "mae": mae.item(), 
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def eval_step(self, batch):
        self.eval()
        images = batch["image"].to(self.device)
        coords = batch["coord"].to(self.device)
        if coords.dim() == 3:
            coords = coords.view(coords.size(0), -1)

        preds = self.model(images)
        loss = self.loss_fn(preds, coords)
        mse = self.mse_metric(preds, coords)
        mae = self.mae_metric(preds, coords)
        return {
            "loss": loss.item(), 
            "mse": mse.item(), 
            "mae": mae.item(), 
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        images = images.to(self.device)
        pred_coords = self.model(images)
        return pred_coords.cpu()
