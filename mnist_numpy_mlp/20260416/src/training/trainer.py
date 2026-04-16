def train(model, dataloader):
    total_loss = 0
    total_acc = 0
    total_size = 0

    for x, y in dataloader:
        batch_size = len(x)
        total_size += batch_size

        loss, acc = model.train_step(x, y)
        total_loss += loss * batch_size
        total_acc += acc * batch_size
    return total_loss / total_size, total_acc / total_size


def evaluate(model, dataloader):
    total_loss = 0
    total_acc = 0
    total_size = 0

    for x, y in dataloader:
        batch_size = len(x)
        total_size += batch_size

        loss, acc = model.eval_step(x, y)
        total_loss += loss * batch_size
        total_acc += acc * batch_size
    return total_loss / total_size, total_acc / total_size


def predict(model, x):
    return model.predict(x)
