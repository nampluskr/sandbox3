import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


class RectIOU:
    def __call__(self, rect1, rect2):
        return self.rect_iou(rect1, rect2)

    def rect_iou(self, rect1, rect2):
        x1 = torch.max(rect1[..., 0], rect2[..., 0])
        y1 = torch.max(rect1[..., 1], rect2[..., 1])
        x2 = torch.min(rect1[..., 2], rect2[..., 2])
        y2 = torch.min(rect1[..., 3], rect2[..., 3])

        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter_area = inter_w * inter_h

        area1 = (rect1[..., 2] - rect1[..., 0]) * (rect1[..., 3] - rect1[..., 1])
        area2 = (rect2[..., 2] - rect2[..., 0]) * (rect2[..., 3] - rect2[..., 1])
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + 1e-8)
        return iou.mean()


class RectRegressor(nn.Module):
    def __init__(self, model, optimizer=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.SmoothL1Loss()
        self.mse_metric = MeanSquaredError().to(self.device)
        self.mae_metric = MeanAbsoluteError().to(self.device)
        self.iou_metric = RectIOU()

    def train_step(self, batch):
        self.train()
        images = batch["image"].to(self.device)
        coords = batch["rect"].to(self.device)

        preds = self.model(images)
        loss = self.loss_fn(preds, coords)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            mse = self.mse_metric(preds, coords)
            mae = self.mae_metric(preds, coords)
            iou = self.iou_metric(preds, coords)
        return {
            "loss": loss.item(), 
            "mse": mse.item(), 
            "mae": mae.item(), 
            "iou": iou.item(),
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def eval_step(self, batch):
        self.eval()
        images = batch["image"].to(self.device)
        coords = batch["rect"].to(self.device)

        preds = self.model(images)
        loss = self.loss_fn(preds, coords)
        mse = self.mse_metric(preds, coords)
        mae = self.mae_metric(preds, coords)
        iou = self.iou_metric(preds, coords)
        return {
            "loss": loss.item(), 
            "mse": mse.item(), 
            "mae": mae.item(), 
            "iou": iou.item(),
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def predict(self, images, sigmoid=False):
        self.eval()
        images = images.to(self.device)
        preds = self.model(images)
        return preds.cpu()


class RectRegressorNorm(nn.Module):
    def __init__(self, model, optimizer=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.SmoothL1Loss()
        self.mse_metric = MeanSquaredError().to(self.device)
        self.mae_metric = MeanAbsoluteError().to(self.device)
        self.iou_metric = RectIOU()

    def train_step(self, batch):
        self.train()
        images = batch["image"].to(self.device)
        coords = batch["rect_norm"].to(self.device)

        logits = self.model(images)
        preds = torch.sigmoid(logits)
        loss = self.loss_fn(preds, coords)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            mse = self.mse_metric(preds, coords)
            mae = self.mae_metric(preds, coords)
            iou = self.iou_metric(preds, coords)
        return {
            "loss": loss.item(), 
            "mse": mse.item(), 
            "mae": mae.item(), 
            "iou": iou.item(),
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def eval_step(self, batch):
        self.eval()
        images = batch["image"].to(self.device)
        coords = batch["rect_norm"].to(self.device)

        logits = self.model(images)
        preds = torch.sigmoid(logits)
        loss = self.loss_fn(preds, coords)
        mse = self.mse_metric(preds, coords)
        mae = self.mae_metric(preds, coords)
        iou = self.iou_metric(preds, coords)
        return {
            "loss": loss.item(), 
            "mse": mse.item(), 
            "mae": mae.item(), 
            "iou": iou.item(),
            "batch_size": images.size(0)
        }

    @torch.no_grad()
    def predict(self, images, sigmoid=False):
        self.eval()
        images = images.to(self.device)
        logits = self.model(images)
        preds = torch.sigmoid(logits)
        return preds.cpu()
