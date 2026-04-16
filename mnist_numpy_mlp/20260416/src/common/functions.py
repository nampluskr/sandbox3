import numpy as np


def identity(x):
    return x


def identity_grad(x):
    return np.ones_like(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


# def sigmoid(x):
#     return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid(x):
    out = np.empty_like(x)
    pos = x > 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    out[neg] = np.exp(x[neg]) / (1.0 + np.exp(x[neg]))
    return out


def sigmoid_grad(x):
    return x * (1 - x)


def softmax(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy(preds, targets):
    if targets.ndim == 1:
        batch_size = preds.shape[0]
        probs = preds[np.arange(batch_size), targets]
    else:   # one-hot labels
        probs = np.sum(preds * targets, axis=1)
    return -np.mean(np.log(probs + 1e-8))


def binary_cross_entropy(preds, targets):
    preds = np.clip(preds, 1e-8, 1 - 1e-8)
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))


def binary_cross_entropy_grad(preds, targets):
    batch_size = preds.shape[0]
    preds = np.clip(preds, 1e-8, 1 - 1e-8)
    return (-(targets / preds) + (1 - targets) / (1 - preds)) / batch_size


def accuracy(preds, targets):
    # preds = softmax(logits)
    if targets.ndim == 2:   # one-hot labels
        targets = targets.argmax(axis=1)
    return (preds.argmax(axis=1) == targets).mean()


def binary_accuracy(preds, targets):
    pred_labels = (preds >= 0.5).astype(int)
    true_labels = targets.astype(int)
    return (pred_labels == true_labels).mean()


def mse(preds, targets):
    return np.mean((preds - targets)**2)


def mse_grad(preds, targets):
    batch_size = preds.shape[0]
    return 2.0 * (preds - targets) / batch_size


def rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets)**2))


def r2_score(preds, targets):
    ss_res = np.sum((preds - targets)**2)
    ss_tot = np.sum((targets - targets.mean())**2)
    return 1.0 - ss_res / (ss_tot + 1e-8)
