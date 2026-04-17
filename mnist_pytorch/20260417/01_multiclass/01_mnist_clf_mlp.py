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
from models.classifier import MulticlassClassifier
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

x_train_np = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test_np = x_test.reshape(-1, 784).astype(np.float32) / 255.0
y_train_np = y_train.astype(np.int64)
y_test_np = y_test.astype(np.int64)

x_train = torch.from_numpy(x_train_np)          # (60000, 784)
y_train = torch.from_numpy(y_train_np)          # (60000, 10)
x_test = torch.from_numpy(x_test_np)            # (10000, 784)
y_test = torch.from_numpy(y_test_np)            # (10000, 10)

train_loader = DataLoader(ImageDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ImageDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

#####################################################################
## Modeling
#####################################################################
torch.manual_seed(SEED)

# model = nn.Sequential(
#     nn.Linear(784, 256),
#     nn.ReLU(),
#     nn.Linear(256, 128),
#     nn.ReLU(),
#     nn.Linear(128, 10),
# )

class MLPModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

model = MLPModel(num_classes=10)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
clf = MulticlassClassifier(model, optimizer, num_classes=10)

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

x = x_test[:NUM_SAMPLES]
y = y_test[:NUM_SAMPLES]
preds = clf.predict(x)

for i in range(NUM_SAMPLES):
    print(f"Target: {y[i]} | Prediction: {preds[i].argmax()}")
