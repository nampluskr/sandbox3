import sys
import os
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CONFIG_DIR = os.path.normpath(os.path.join(ROOT_DIR, "configs"))
SOURCE_DIR = os.path.normpath(os.path.join(ROOT_DIR, "src"))

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

#################################################################
## 1. Hyperparameters
#################################################################
from common.config import load_config
from common.mnist import load_images
from common.functions import sigmoid, sigmoid_grad, mse
from common.modules import Linear, Sigmoid, Sequential
from training.optimizers import SGD

config = load_config(CONFIG_DIR, "autoencoder.yaml")

SEED = config["seed"]
DATASET_DIR = config["dataset_dir"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = float(config["learning_rate"])
NUM_EPOCHS = config["num_epochs"]
NUM_SAMPLES = config["num_samples"]

np.random.seed(SEED)

#################################################################
## 2. Data loading and preprocessing
#################################################################
x_train = load_images(DATASET_DIR, "train")
x_test = load_images(DATASET_DIR, "test")

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

#################################################################
## 3. Modeling (Autoencoder with 2D latent space)
#################################################################
np.random.seed(SEED)

encoder = Sequential(
    Linear(784, 256),
    Sigmoid(),
    Linear(256, 128),
    Sigmoid(),
    Linear(128, 2)
)

decoder = Sequential(
    Linear(2, 128),
    Sigmoid(),
    Linear(128, 256),
    Sigmoid(),
    Linear(256, 784)
)
autoencoder = Sequential(
    encoder,
    decoder
)
optimizer = SGD(autoencoder, lr=LEARNING_RATE)

#################################################################
## 4. Training
#################################################################
print(f"\n>> Autoencoder Training (2D Latent Space):")

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = 0
    total_size = 0

    indices = np.random.permutation(len(x_train))

    for idx in range(0, len(x_train), BATCH_SIZE):
        x = x_train[indices[idx: idx + BATCH_SIZE]]
        batch_size = len(x)
        total_size += batch_size

        # Forward
        latent = encoder(x)
        logits = decoder(latent)
        recon = sigmoid(logits)
        loss = mse(recon, x)

        # Backward
        dout = 2 * (recon - x) / batch_size
        dout = sigmoid_grad(dout) * dout
        dout = decoder.backward(dout)
        encoder.backward(dout)

        # Update weights
        optimizer.step()

        total_loss += loss * batch_size

    print(f"[{epoch:>2}/{NUM_EPOCHS}] loss: {total_loss/total_size:.4f}")

#################################################################
## 5. Evaluation
#################################################################
print(f"\n>> Autoencoder Evaluation:")

total_loss = 0.0
total_size = 0

for idx in range(0, len(x_test), BATCH_SIZE):
    x = x_test[idx:idx + BATCH_SIZE]
    batch_size = x.shape[0]
    total_size += batch_size

    latent = encoder(x)
    logits = decoder(latent)
    recon = sigmoid(logits)
    loss = mse(recon, x)

    total_loss += loss * batch_size

print(f"Test Loss: {total_loss/total_size:.4f}")

#################################################################
## 6. Latent Space & Reconstruction
#################################################################
print(f"\n>> Latent Space & Reconstruction Samples:")

x = x_test[:NUM_SAMPLES]  # (10, 784)

# Encode
latent = encoder(x)

print("Latent vectors (2D):")
for i in range(NUM_SAMPLES):
    print(f"Sample {i+1}: [{latent[i, 0]:.3f}, {latent[i, 1]:.3f}]")

# # Decode
# h4 = np.dot(latent, w4) + b4
# a4 = sigmoid(h4)
# h5 = np.dot(a4, w5) + b5
# a5 = sigmoid(h5)
# h6 = np.dot(a5, w6) + b6
# recon = sigmoid(h6)  # (10, 784)

# for i in range(NUM_SAMPLES):
#     # reshape을 f-string 밖에서 수행
#     original_img = x_sample[i].reshape(28, 28)
#     recon_img = recon[i].reshape(28, 28)

#     print(f"\nSample {i+1}")
#     print(f"Original (row 14): {original_img[14, :] * 255:.1f}")
#     print(f"Recon    (row 14): {recon_img[14, :] * 255:.1f}")
