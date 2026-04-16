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
from common.functions import sigmoid, sigmoid_grad
from common.functions import identity, mse, r2_score

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

w1 = np.random.randn(784, 256)
b1 = np.zeros(256)
w2 = np.random.randn(256, 128)
b2 = np.zeros(128)
w3 = np.random.randn(128, 1)
b3 = np.zeros(1)

#################################################################
## 4. Training
#################################################################
print(f"\n>> Training:")

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = 0
    total_acc = 0
    total_size = 0

    indices = np.random.permutation(len(x_train))

    for idx in range(0, len(x_train), BATCH_SIZE):
        x = x_train[indices[idx: idx + BATCH_SIZE]]
        y = y_train[indices[idx: idx + BATCH_SIZE]]
        batch_size = len(x)
        total_size += batch_size

        # Forward propagation
        z1 = np.dot(x, w1) + b1                     # (N, 256)
        a1 = sigmoid(z1)                            # (N, 256)
        z2 = np.dot(a1, w2) + b2                    # (N, 128)
        a2 = sigmoid(z2)                            # (N, 128)
        z3 = np.dot(a2, w3) + b3                    # (N, 10)

        preds = identity(z3)                        # (N, 1)
        loss = mse(preds, y)                        # (N, 1), (N, 1)
        acc = r2_score(preds, y)                    # (N, 1), (N, 1)

        # Backward propagation
        # dout = mse_grad(y_preds, y)               # (N, 1)
        # grad_z3 = identity_grad(y_preds) * dout   # (N, 1)
        grad_z3 = 2 * (preds - y) / batch_size      # (N, 1)
        grad_w3 = np.dot(a2.T, grad_z3)             # (128, 1)
        grad_b3 = np.sum(grad_z3, axis=0)           # (1, )

        grad_a2 = np.dot(grad_z3, w3.T)             # (N, 128)
        grad_z2 = sigmoid_grad(a2) * grad_a2        # (N, 128)
        grad_w2 = np.dot(a1.T, grad_z2)             # (256, 128)
        grad_b2 = np.sum(grad_z2, axis=0)           # (128,)

        grad_a1 = np.dot(grad_z2, w2.T)             # (N, 256)
        grad_z1 = sigmoid_grad(a1) * grad_a1        # (N, 256)
        grad_w1 = np.dot(x.T, grad_z1)              # (784, 256)
        grad_b1 = np.sum(grad_z1, axis=0)           # (256,)

        # Update weights
        w1 -= LEARNING_RATE * grad_w1
        b1 -= LEARNING_RATE * grad_b1
        w2 -= LEARNING_RATE * grad_w2
        b2 -= LEARNING_RATE * grad_b2
        w3 -= LEARNING_RATE * grad_w3
        b3 -= LEARNING_RATE * grad_b3

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

    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3) + b3

    preds = identity(z3)
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

z1 = np.dot(x, w1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, w2) + b2
a2 = sigmoid(z2)
z3 = np.dot(a2, w3) + b3

preds = identity(z3)

for i in range(NUM_SAMPLES):
    raw = preds[i, 0]
    pred_label = int(np.round(np.clip(raw * 9.0, 0, 9)))
    true_label = int(np.round(y[i, 0] * 9.0))
    print(f"Target: {true_label} | Prediction: {pred_label} (raw: {np.clip(raw * 9.0, 0, 9):.4f})")
