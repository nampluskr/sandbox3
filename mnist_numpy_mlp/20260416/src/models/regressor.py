from common.functions import identity, mse, r2_score


class Regressor:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y):
        logits = self.model(x)
        preds = identity(logits)
        loss = mse(preds, y)
        acc = r2_score(preds, y)

        dout = 2 * (preds - y) / x.shape[0]
        self.model.backward(dout)
        self.optimizer.step()
        return loss, acc

    def eval_step(self, x, y):
        logits = self.model(x)
        preds = identity(logits)
        loss = mse(preds, y)
        acc = r2_score(preds, y)
        return loss, acc

    def predict(self, x):
        logits = self.model(x)
        preds = identity(logits)
        return preds
