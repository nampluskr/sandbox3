import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import R2Score


class Regressor:
    def __init__(self, model, optimizer=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.r2_metric = R2Score().to(self.device)

    def train_step(self, batch):
        self.model.train()
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        r2 = self.r2_metric(logits, labels)
        return {"loss": loss.item(), "r2": r2.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        r2 = self.r2_metric(logits, labels)
        return {"loss": loss.item(), "r2": r2.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        images = images.to(self.device)
        preds = self.model(images)
        return preds.cpu()
