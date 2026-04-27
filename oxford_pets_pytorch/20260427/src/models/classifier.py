import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import Accuracy, BinaryAccuracy


class MulticlassClassifier(nn.Module):
    def __init__(self, model, num_classes, optimizer=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)

    def train_step(self, batch):
        self.train()
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = self.acc_metric(logits, labels)
        return {"loss": loss.item(), "acc": acc.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def eval_step(self, batch):
        self.eval()
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = self.acc_metric(logits, labels)
        return {"loss": loss.item(), "acc": acc.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        images = images.to(self.device)
        logits = self.model(images)
        preds = torch.softmax(logits, dim=1)
        return preds.cpu()


class BinaryClassifier(nn.Module):
    def __init__(self, model, optimizer=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc_metric = BinaryAccuracy().to(self.device)

    def train_step(self, batch):
        self.model.train()
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = self.acc_metric(logits, labels)
        return {"loss": loss.item(), "acc": acc.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        acc = self.acc_metric(logits, labels)
        return {"loss": loss.item(), "acc": acc.item(), "batch_size": images.size(0)}

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        images = images.to(self.device)
        logits = self.model(images)
        preds = torch.sigmoid(logits)
        return preds.cpu()
