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
from models.autoencoder import AutoEncoder
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

class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, output_padding=0)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 4, 4)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = self.deconv3(x)
        return torch.sigmoid(x)

encoder = Encoder(latent_dim=2)
decoder = Decoder(latent_dim=2)
ae = AutoEncoder(encoder, decoder)

#####################################################################
## Training
#####################################################################
print("\n>> Training:")

# for epoch in range(1, NUM_EPOCHS + 1):
#     train_results = train(clf, train_loader)
#     print(f"[{epoch:>2}/{NUM_EPOCHS}] {train_results['info']}")

history = fit(ae, train_loader, num_epochs=NUM_EPOCHS, valid_loader=test_loader)

#####################################################################
## Evaluation
#####################################################################
print("\n>> Evalutaion:")

test_results = evaluate(ae, test_loader)
print(test_results["info"])

# #################################################################
# ## Prediction
# #################################################################
# print(f"\n>> Prediction:")

# x = x_test[:NUM_SAMPLES]
# y = y_test[:NUM_SAMPLES]
# preds = clf.predict(x)

# for i in range(NUM_SAMPLES):
#     print(f"Target: {y[i]} | Prediction: {preds[i].argmax()}")
