import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from shapely.geometry import Polygon


class RectIOU:
    def __call__(self, rect1, rect2):
        return self.rect_iou(rect1, rect2)

    def rect_iou(self, rect1, rect2):
        if rect1.size(-1) == 4:
            x1 = torch.max(rect1[..., 0], rect2[..., 0])
            y1 = torch.max(rect1[..., 1], rect2[..., 1])
            x2 = torch.min(rect1[..., 2], rect2[..., 2])
            y2 = torch.min(rect1[..., 3], rect2[..., 3])
            area1 = (rect1[..., 2] - rect1[..., 0]) * (rect1[..., 3] - rect1[..., 1])
            area2 = (rect2[..., 2] - rect2[..., 0]) * (rect2[..., 3] - rect2[..., 1])

        elif rect1.size(-1) == 8:
            x1 = torch.max(rect1[..., 0], rect2[..., 0])
            y1 = torch.max(rect1[..., 1], rect2[..., 1])
            x2 = torch.min(rect1[..., 4], rect2[..., 4])
            y2 = torch.min(rect1[..., 5], rect2[..., 5])
            area1 = (rect1[..., 4] - rect1[..., 0]) * (rect1[..., 5] - rect1[..., 1])
            area2 = (rect2[..., 4] - rect2[..., 0]) * (rect2[..., 5] - rect2[..., 1])

        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter_area = inter_w * inter_h

        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + 1e-8)
        return iou.mean()


class PolyIOU:
    def __call__(self, poly1, poly2):
        return self.poly_iou(poly1, poly2)

    def poly_iou(self, poly1, poly2):
        assert poly1.shape == poly2.shape
        assert poly1.shape[-1] == 8

        device = poly1.device
        N = poly1.view(-1, 8).size(0)
        poly1 = poly1.view(N, 8).cpu().numpy()
        poly2 = poly2.view(N, 8).cpu().numpy()

        iou_values = []
        for i in range(N):
            try:
                p1 = Polygon(poly1[i].reshape(4, 2))
                p2 = Polygon(poly2[i].reshape(4, 2))

                if not p1.is_valid:
                    p1 = p1.buffer(0)
                if not p2.is_valid:
                    p2 = p2.buffer(0)

                inter_area = p1.intersection(p2).area
                union_area = p1.union(p2).area
                iou = inter_area / (union_area + 1e-8)

            except Exception:
                iou = 0.0

            iou_values.append(iou)
        return torch.tensor(iou_values, device=device, dtype=torch.float32).mean()


class Regressor(nn.Module):
    def __init__(self, model, optimizer=None, device=None, use_sigmoid=True, iou=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)
        self.mse_metric = MeanSquaredError().to(self.device)
        self.mae_metric = MeanAbsoluteError().to(self.device)
        self.iou_metric = iou or RectIOU()
        self.use_sigmoid = use_sigmoid

    def forward(self, batch):
        images = batch["image"].to(self.device)
        logits = self.model(images)

        if self.use_sigmoid:
            coords = batch["coord_norm"].to(self.device)
            preds = torch.sigmoid(logits)
        else:
            coords = batch["coord"].to(self.device)
            preds = logits
        return preds, coords

    def train_step(self, batch):
        self.train()
        preds, coords = self.forward(batch)
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
            "batch_size": preds.size(0)
        }

    @torch.no_grad()
    def eval_step(self, batch):
        self.eval()
        preds, coords = self.forward(batch)
        loss = self.loss_fn(preds, coords)
        mse = self.mse_metric(preds, coords)
        mae = self.mae_metric(preds, coords)
        iou = self.iou_metric(preds, coords)
        return {
            "loss": loss.item(),
            "mse": mse.item(),
            "mae": mae.item(),
            "iou": iou.item(),
            "batch_size": preds.size(0)
        }

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        images = images.to(self.device)
        logits = self.model(images)
        preds = torch.sigmoid(logits) if self.use_sigmoid else logits
        return preds.cpu()
