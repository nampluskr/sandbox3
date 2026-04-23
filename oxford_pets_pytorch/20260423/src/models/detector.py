import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.detection import MeanAveragePrecision
from typing import List, Dict, Any
import torchvision.tv_tensors as tv_tensors


class Detector(nn.Module):
    def __init__(self, model, num_classes, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.bbox_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_weight = 0.5

    def forward(self, images, targets):
        outputs = self.model(images, targets)
        if isinstance(outputs, dict):
            return outputs

        H, W = images.shape[-2:]
        gt_bboxes = torch.stack([t["boxes"][0] for t in targets]) / torch.tensor([W, H, W, H]).to(self.device)
        gt_labels = torch.stack([t["labels"][0] for t in targets])

        pred_bboxes, pred_logits = outputs
        loss_bbox = self.bbox_loss(pred_bboxes, gt_bboxes)
        loss_cls = self.cls_loss(pred_logits, gt_labels)

        return {
            "loss_bbox": loss_bbox * self.bbox_weight,
            "loss_cls": loss_cls,
        }

    def train_step(self, batch):
        self.model.train()
        images = batch["image"].to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch["target"]]

        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            **{f"{k}": v.item() for k, v in loss_dict.items()},
            "batch_size": images.size(0),
        }

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        images = batch["image"].to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch["target"]]

        loss_dict = self.forward(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        return {
            "total": total_loss.item(),
            **{f"{k}": v.item() for k, v in loss_dict.items()},
            "batch_size": images.size(0),
        }

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        images = batch["image"].to(self.device)
        outputs = self.model(images)
        return outputs
