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
from common.mnist import load_images, load_labels, one_hot
from common.modules import Linear, ReLU, Sequential
from common.dataloader import Dataloader
from training.optimizers import Adam
from training.trainer import train, evaluate, predict
from models.classifier import MulticlassClassifier

config = load_config(CONFIG_DIR, "multiclass.yaml")

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

y_train = one_hot(y_train, num_classes=10).astype(np.float32)   # (60000, 10)
y_test = one_hot(y_test, num_classes=10).astype(np.float32)     # (10000, 10)

#################################################################
## 3. Modeling
#################################################################
np.random.seed(SEED)

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
)
model.layers[0].w *= np.sqrt(2 / 784)   # He(Kaiming) initialization (for ReLU, LeakyReLU)
model.layers[2].w *= np.sqrt(2 / 256)   # He(Kaiming) initialization (for ReLU, LeakyReLU)
model.layers[4].w *= np.sqrt(2 / 128)   # He(Kaiming) initialization (for ReLU, LeakyReLU)

optimizer = Adam(model, lr=0.001)
clf = MulticlassClassifier(model, optimizer)

#################################################################
## 4. Training
#################################################################
print(f"\n>> Training:")

train_loader = Dataloader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Dataloader(x_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train(clf, train_loader)
    print(f"[{epoch:>2}/{NUM_EPOCHS}] loss:{train_loss:.3f} acc:{train_acc:.3f}")

#################################################################
## 5. Evaluaton
#################################################################
print(f"\n>> Evaluation:")

test_loss, test_acc = evaluate(clf, test_loader)
print(f"loss:{test_loss:.3f} acc:{test_acc:.3f}")

#################################################################
## 6. Prediction
#################################################################
print(f"\n>> Prediction:")

x = x_test[:NUM_SAMPLES]
y = y_test[:NUM_SAMPLES]

preds = predict(clf, x)

for i in range(NUM_SAMPLES):
    print(f"Target: {y[i].argmax()} | Prediction: {preds[i].argmax()}")
