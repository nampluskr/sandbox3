import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
CONFIG_DIR = os.path.normpath(os.path.join(ROOT_DIR, "configs"))
SOURCE_DIR = os.path.normpath(os.path.join(ROOT_DIR, "src"))

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

#####################################################################
## Libraries and Hyperparameters
#####################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common.config import load_config
from common.mnist import load_images, load_labels, one_hot, ImageDataset
from models.classifier import BinaryClassifier
from training.trainer import train, evaluate, fit

config = load_config(CONFIG_DIR, "default.yaml")

SEED = config["seed"]
DATASET_DIR = config["dataset_dir"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = float(config["learning_rate"])
NUM_EPOCHS = config["num_epochs"]
NUM_SAMPLES = config["num_samples"]

#####################################################################
## Data Loading
#####################################################################

x_train = load_images(DATASET_DIR, "train")     # (60000, 28, 28)
y_train = load_labels(DATASET_DIR, "train")     # (60000,)
x_test = load_images(DATASET_DIR, "test")       # (10000, 28, 28)
y_test = load_labels(DATASET_DIR, "test")       # (10000,)

x_train_np = x_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
x_test_np = x_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
y_train_np = (y_train % 2).astype(np.float32).reshape(-1, 1)
y_test_np = (y_test % 2).astype(np.float32).reshape(-1, 1)

x_train = torch.from_numpy(x_train_np)          # (60000, 784)
y_train = torch.from_numpy(y_train_np)          # (60000, 1)
x_test = torch.from_numpy(x_test_np)            # (10000, 784)
y_test = torch.from_numpy(y_test_np)            # (10000, 1)

train_loader = DataLoader(ImageDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ImageDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

#####################################################################
## Modeling
#####################################################################
torch.manual_seed(SEED)

# model = nn.Sequential(
#     nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2),
#     nn.Flatten(),
#     nn.Dropout(p=0.5),
#     nn.Linear(32 * 7 * 7, 1),
# )

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 7 * 7, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
clf = BinaryClassifier(model, optimizer)

#####################################################################
## Training
#####################################################################
print("\n>> Training:")

# for epoch in range(1, NUM_EPOCHS + 1):
#     train_results = train(clf, train_loader)
#     print(f"[{epoch:>2}/{NUM_EPOCHS}] {train_results['info']}")

history = fit(clf, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)

#####################################################################
## Evaluation
#####################################################################
print("\n>> Evalutaion:")

test_results = evaluate(clf, test_loader)
print(test_results["info"])

#################################################################
## Prediction
#################################################################
print(f"\n>> Prediction:")

label_str = {0: "even", 1: "odd"}
x = x_test[:NUM_SAMPLES]
y = y_test[:NUM_SAMPLES]
preds = clf.predict(x)

for i in range(NUM_SAMPLES):
    pred_label = int(preds[i, 0] >= 0.5)
    true_label = int(y[i, 0])
    print(f"Target: {true_label}({label_str[true_label]:<4}) | "
          f"Prediction: {pred_label}({label_str[pred_label]:<4}) "
          f"(prob_odd: {preds[i, 0]:.3f})")
