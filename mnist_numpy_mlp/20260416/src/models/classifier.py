from common.functions import softmax, cross_entropy, accuracy
from common.functions import sigmoid, binary_cross_entropy, binary_accuracy


class MulticlassClassifier:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y):
        logits = self.model(x)
        preds = softmax(logits)
        loss = cross_entropy(preds, y)
        acc = accuracy(preds, y)

        dout = (preds - y) / x.shape[0]
        self.model.backward(dout)
        self.optimizer.step()
        return loss, acc

    def eval_step(self, x, y):
        logits = self.model(x)
        preds = softmax(logits)
        loss = cross_entropy(preds, y)
        acc = accuracy(preds, y)
        return loss, acc

    def predict(self, x):
        logits = self.model(x)
        preds = softmax(logits)
        return preds


class BinaryClassifier:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y):
        logits = self.model(x)
        preds = sigmoid(logits)
        loss = binary_cross_entropy(preds, y)
        acc = binary_accuracy(preds, y)

        dout = (preds - y) / x.shape[0]
        self.model.backward(dout)
        self.optimizer.step()
        return loss, acc

    def eval_step(self, x, y):
        logits = self.model(x)
        preds = sigmoid(logits)
        loss = binary_cross_entropy(preds, y)
        acc = binary_accuracy(preds, y)
        return loss, acc

    def predict(self, x):
        logits = self.model(x)
        preds = sigmoid(logits)
        return preds
