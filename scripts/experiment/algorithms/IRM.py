import torch
from torch.autograd import grad





# def example_1(n=10000, d=1, env=1):
#     x = torch.randn(n, d) * env
#     y = x + torch.randn(n, d) * env
#     z = y + torch.randn(n, d)
#     return torch.cat((x, z), 1), y.sum(1, keepdim=True)


# phi = torch.nn.Parameter(torch.ones(2, 1))
# dummy_w = torch.nn.Parameter(torch.Tensor([1.0]))

# opt = torch.optim.SGD([phi], lr=1e-3)
# mse = torch.nn.MSELoss(reduction="none")

# environments = [example_1(env=0.1), example_1(env=1.0)]

# for iteration in range(50000):
#     error = 0
#     penalty = 0
#     for x_e, y_e in environments:
#         p = torch.randperm(len(x_e))
#         error_e = mse(x_e[p] @ phi * dummy_w, y_e[p])
#         penalty += compute_penalty(error_e, dummy_w)
#         error += error_e.mean()

#     opt.zero_grad()
#     (1e-5 * error + penalty).backward()
#     opt.step()

#     if iteration % 1000 == 0:
#         print(phi)


# problem with the dimensions for losses, not the grad!!!!!
def compute_penalty(losses, dummy_w):
    g1 = grad(losses[0::2].mean(), dummy_w, create_graph=True)[0]
    g2 = grad(losses[1::2].mean(), dummy_w, create_graph=True)[0]
    return (g1 * g2).sum()

def IRM(
    model, train_loader, optimizer, loss_func, epoch, config, dummy_w, ERM_epochs=0, lam=1e-5, 
):


# Need to define dummy_w and think about it!!!!
# dummy_w is R^num_outputs, dummy_w = (1,1,1,1,1,1)

    """
    Trains a classifier for an epoch and returns the train loss and train accuracy for that epoch
    Uses the IRM algorithm 
    """

    train_loss = train_acc = 0.0

    # Re-define loss function to set reduction to none
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    """
    IRM loss
    """
    batches_per_epoch = (
        len(train_loader[0].dataset) * len(train_loader) // (config.train_batch_size)
    )
    train_loader_iter = [iter(loader) for loader in train_loader]

    count = 0
    while count < batches_per_epoch:
        # do one batch from each environment
        error = 0
        penalty = 0
        # sample batch from each environment
        # Note: the dataset is covered in format batch1_e1,.., batch1_eE, batch2_e1,..
        for loader in train_loader_iter:
            batch = next(loader, None)
            n_envs = 0
            if batch is not None:
                count += 1
                img, labels = batch
                img, labels = img.to(config.device), labels.to(config.device)
                predictions = model(img)
                error_e = loss_func(predictions * dummy_w, labels)
                # dim_output = predictions.size(dim=-1)
                # dummy_w = torch.nn.Parameter(torch.eye((dim_output))).to(config.device)
                # error_e = loss_func(torch.matmul(predictions, dummy_w), labels)
                penalty += compute_penalty(error_e, dummy_w)
                error += error_e.mean()
                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == labels
                train_acc += correct.sum()

            n_envs+=1

        # optimize the worst loss
        penalty /= n_envs
        error /= n_envs
        optimizer.zero_grad()
        # if epoch % 1 == 0 and count < 5:
        #     print('lam is ', lam)
        #     print('error is ', error)
        #     print('penalty is ', penalty)
        #     print(f"train acc is {train_acc}/{len(train_loader[0].dataset) * len(train_loader)}")
        loss = error + lam * penalty
        # loss = error
        loss.backward()
        optimizer.step()

        #train_loss += loss
        train_loss += error.item() # making it this to make it more interpretable as loss includes the penalty and not just the cross entropy

    train_loss /= len(train_loader[0].dataset) * len(train_loader)
    train_acc /= len(train_loader[0].dataset) * len(train_loader)

    return train_loss, train_acc
