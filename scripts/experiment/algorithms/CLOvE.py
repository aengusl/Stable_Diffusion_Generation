## implementation from https://github.com/TjuJianyu/RFC/blob/master/src/algorithms/CLOvE.py

import torch
import torch.nn.functional as F

# problem with the dimensions for losses, not the grad!!!!!
def compute_penalty(logits, y, kernel_scale=0.4, kernel='laplacian'):
    if logits.shape[1] > 1: 
        y_hat = logits.argmax(dim=1).flatten()
        probs = F.softmax(logits,dim=1).flatten()

    else:
        y_hat = logits.flatten() > 0
        probs = F.sigmoid(logits).flatten()

    
    c = ~(  y_hat ^ y  )
    c = c.detach().float()

    confidence = torch.ones(len(y_hat)).cuda()
    confidence[y_hat] = 1-probs[y_hat]
    confidence[~y_hat] = probs[~y_hat]

    k = (-(confidence.view(-1,1)-confidence).abs() / kernel_scale).exp()
    conf_diff = (c - confidence).view(-1,1)  * (c -confidence) 

    res = conf_diff * k
    return res.sum() / (len(logits)**2)

def CLOvE(
    model, train_loader, optimizer, loss_func, epoch, config, ERM_epochs=0, penalty_weight=1e-5
):


# Need to define dummy_w and think about it!!!!
# dummy_w is R^num_outputs, dummy_w = (1,1,1,1,1,1)

    """
    Trains a classifier for an epoch and returns the train loss and train accuracy for that epoch
    Uses the IRM algorithm 
    """

    train_loss = train_acc = 0.0

    """
    IRM loss
    """
    batches_per_epoch = (
        len(train_loader[0].dataset) * len(train_loader) // (config.train_batch_size)
    )
    train_loader_iter = [iter(loader) for loader in train_loader]
    list_losses = []
    pred_list = []
    label_list = []

    count = 0
    while count < batches_per_epoch:
        # do one batch from each environment
        avg_penalty = 0.
        avg_loss = 0.
        n_envs = 0

        # sample batch from each environment
        # Note: the dataset is covered in format batch1_e1,.., batch1_eE, batch2_e1,..

        # need to collect errors for each environment separately, in a list
        for loader in train_loader_iter:
            batch = next(loader, None)
            if batch is not None:
                count += 1
                img, labels = batch
                img, labels = img.to(config.device), labels.to(config.device)
                predictions = model(img)
                avg_loss += loss_func(predictions, labels)
                avg_penalty += compute_penalty(predictions.data, labels)
                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == labels
                train_acc += correct.sum()
                n_envs+=1

        avg_penalty /= n_envs
        avg_loss /= n_envs
            

        # optimize the worst loss
        optimizer.zero_grad()
        loss = avg_loss + penalty_weight*avg_penalty
        loss.backward()
        optimizer.step()

        train_loss += avg_loss

        count += 1

    train_loss /= len(train_loader[0].dataset) * len(train_loader)
    train_acc /= len(train_loader[0].dataset) * len(train_loader)

    return train_loss, train_acc
