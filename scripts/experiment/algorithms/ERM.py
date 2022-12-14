import torch


def ERM(model, train_loader, optimizer, loss_func, epoch, config):

    """
    Trains and ERM classifier for an epoch and returns the train loss and train accuracy for that epoch
    """

    train_loss = train_acc = 0.0
    for (img, labels) in train_loader:
        img, labels = img.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        predictions = model(img)
        loss = loss_func(predictions, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct = torch.argmax(predictions.data, 1) == labels
        train_loss += loss
        train_acc += correct.sum()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    return train_loss, train_acc
