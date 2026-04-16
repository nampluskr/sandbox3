import sys
import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CONFIG_DIR = os.path.normpath(os.path.join(ROOT_DIR, "configs"))
SOURCE_DIR = os.path.normpath(os.path.join(ROOT_DIR, "src"))

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

#################################################################
## 1. Hyperparameters
#################################################################
import numpy as np

from common.config import load_config
from common.mnist import load_images, load_labels
from common.functions import identity, mse, r2_score
from common.modules import Linear, Sigmoid, Sequential
from common.dataloader import Dataloader
from training.optimizers import SGD

config = load_config(CONFIG_DIR, "regression.yaml")

SEED = config["seed"]
DATASET_DIR = config["dataset_dir"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = float(config["learning_rate"])
NUM_EPOCHS = config["num_epochs"]
NUM_SAMPLES = config["num_samples"]

#################################################################
## 2. Data loading and preprocessing
#################################################################
x_train = load_images(DATASET_DIR, "train")                     # (60000, 28, 28)
y_train = load_labels(DATASET_DIR, "train")                     # (60000,)
x_test = load_images(DATASET_DIR, "test")                       # (10000, 28, 28)
y_test = load_labels(DATASET_DIR, "test")                       # (10000,)

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0   # (60000, 784)
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0     # (10000, 784)

y_train = y_train.astype(np.float32).reshape(-1, 1) / 9.0       # (60000, 1)
y_test = y_test.astype(np.float32).reshape(-1, 1) / 9.0         # (10000, 1)

#################################################################
## 3. Modeling
#################################################################
np.random.seed(SEED)

model = Sequential(
    Linear(784, 256),
    Sigmoid(),
    Linear(256, 128),
    Sigmoid(),
    Linear(128, 1),
)
optimizer = SGD(model, lr=LEARNING_RATE)

#################################################################
## 4. Training
#################################################################
print(f"\n>> Training:")

train_loader = Dataloader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Dataloader(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = 0
    total_acc = 0
    total_size = 0

    for x, y in train_loader:
        batch_size = len(x)
        total_size += batch_size

        # Forward propagation
        logits = model(x)
        preds = identity(logits)
        loss = mse(preds, y)
        acc = r2_score(preds, y)

        # Backward propagation
        dout = 2 * (preds - y) / batch_size
        model.backward(dout)

        # Update weights
        optimizer.step()

        total_loss += loss * batch_size
        total_acc += acc * batch_size

    print(f"[{epoch:>2}/{NUM_EPOCHS}] loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

#################################################################
## 5. Evaluaton
#################################################################
print(f"\n>> Evaluation:")

total_loss = 0.0
total_acc = 0.0
total_size = 0

for idx in range(0, len(x_test), BATCH_SIZE):
    x = x_test[idx:idx + BATCH_SIZE]
    y = y_test[idx:idx + BATCH_SIZE]
    batch_size = x.shape[0]
    total_size += batch_size

    logits = model(x)
    preds = identity(logits)
    loss = mse(preds, y)
    acc = r2_score(preds, y)

    total_loss += loss * batch_size
    total_acc += acc * batch_size

print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

#################################################################
## 6. Prediction
#################################################################
print(f"\n>> Prediction:")

x = x_test[:NUM_SAMPLES]
y = y_test[:NUM_SAMPLES]

logits = model(x)
preds = identity(logits)

for i in range(NUM_SAMPLES):
    raw = preds[i, 0]
    pred_label = int(np.round(np.clip(raw * 9.0, 0, 9)))
    true_label = int(np.round(y[i, 0] * 9.0))
    print(f"Target: {true_label} | Prediction: {pred_label} (raw: {np.clip(raw * 9.0, 0, 9):.4f})")
