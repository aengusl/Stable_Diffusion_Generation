import torch


def gDRO(model, train_loader, optimizer, loss_func, epoch, config, ERM_epochs=0):

    """
    Trains a classifier for an epoch and returns the train loss and train accuracy for that epoch
    Uses the group DRO algorithm as described in https://arxiv.org/abs/1911.08731
    """

    train_loss = train_acc = 0.0
    if epoch < ERM_epochs:

        batches_per_epoch = (
            len(train_loader[0].dataset)
            * len(train_loader)
            // (config.train_batch_size)
        )
        train_loader_iter = [iter(loader) for loader in train_loader]
        list_losses = []
        pred_list = []
        label_list = []

        count = 0
        while count < batches_per_epoch:
            # do one batch from each environment
            image_list = []
            label_list = []

            # sample batch from each environment
            for loader in train_loader_iter:
                batch = next(loader, None)
                if batch is not None:

                    img, labels = batch
                    img, labels = img.to(config.device), labels.to(config.device)
                    image_list.append(img)
                    label_list.append(labels)

            concat_image = torch.cat(tuple(image_list), 0)
            concat_labels = torch.cat(tuple(label_list), 0)
            # optimize the worst loss
            optimizer.zero_grad()
            predictions = model(concat_image)
            loss = loss_func(predictions, concat_labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == concat_labels
            train_loss += loss
            train_acc += correct.sum()
            count += 1

    else:

        """
        DRO loss
        """
        batches_per_epoch = (
            len(train_loader[0].dataset)
            * len(train_loader)
            // (config.train_batch_size)
        )
        train_loader_iter = [iter(loader) for loader in train_loader]
        list_losses = []
        pred_list = []
        label_list = []

        count = 0
        while count < batches_per_epoch:
            # do one batch from each environment
            image_list = []
            label_list = []

            # sample batch from each environment
            for loader in train_loader_iter:
                batch = next(loader, None)
                n_envs = 0
                if batch is not None:

                    img, labels = batch
                    img, labels = img.to(config.device), labels.to(config.device)
                    image_list.append(img)
                    label_list.append(labels)
                    count += 1

            # find the worst loss
            with torch.no_grad():
                n_envs = len(train_loader)
                losses = []
                env_batch_size = config.train_batch_size // n_envs
                for i in range(n_envs):
                    predictions = model(image_list[i])
                    #print('dim of predictions is', predictions.shape)
                    #print('dim of labes is', label_list[i].shape)
                    loss = loss_func(predictions, label_list[i])
                    losses.append(loss)
                max_index = losses.index(max(losses))

            image = image_list[max_index]
            label = label_list[max_index]

            # optimize the worst loss
            optimizer.zero_grad()
            predictions = model(image)
            loss = loss_func(predictions, label)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == label
            train_loss += loss
            train_acc += correct.sum()

            count += 1

    train_loss /= len(train_loader[0].dataset) * len(train_loader)
    train_acc /= len(train_loader[0].dataset) * len(train_loader)

    return train_loss, train_acc
