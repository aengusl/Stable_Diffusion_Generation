import math
import os
from datetime import datetime
from fnmatch import fnmatch

import torch
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from cifar_resnet import resnet18
from config_classifier import get_config
from utils import set_seed
from get_the_data_original import get_train_valid_test_loaders

config = get_config()
EXP_NAME = f"aug={config.augment_data}_data={config.dataset}_model={config.classifier_arch}_lr={config.lr}_seed={config.seed}"
config.save_path += EXP_NAME + "/{:%Y_%m_%d_%H_%M_%S_%f}/".format(datetime.now())
os.makedirs(config.save_path, exist_ok=True)
set_seed(config.seed)


if config.wandb:
    import wandb

    wandb.init(
        name=EXP_NAME,
        project=config.wandb_project,
        config=config.to_dict(),
    )


# ImageFolder assumes that differently labeled images lie in different subfolders
# e.g.
#         root/dog/xxx.png
#         root/dog/xxy.png
#         root/dog/[...]/xxz.png
#
#         root/cat/123.png
#         root/cat/nsdf3.png
#         root/cat/[...]/asd932_.png

data = datasets.ImageFolder(
    root=config.fake_dataset_path,
    transform=transforms.transforms.ToTensor(),
)

# split into train and validation set
train_size, valid_size = math.floor(len(data) * (1 - config.val_size)), math.ceil(
    len(data) * config.val_size
)
train_set, val_set = torch.utils.data.random_split(data, [train_size, valid_size])


train_data_concat = torch.cat([d[0] for d in DataLoader(train_set)])
mean = train_data_concat.mean(dim=[0, 2, 3])
std = train_data_concat.std(dim=[0, 2, 3])


# Construct transforms
train_transforms = [
    transforms.transforms.RandomCrop(size=(32, 32), padding=4),
    transforms.transforms.RandomHorizontalFlip(),
    transforms.transforms.ToTensor(),
    transforms.transforms.Normalize(mean, std),
]

test_transform = transforms.transforms.Compose(
    [
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(mean, std),
    ]
)


train_data_loader = DataLoader(
    train_set,
    batch_size=config.train_batch_size,
    shuffle=True,
    pin_memory=False,
    drop_last=False,
)
val_data_loader = (
    DataLoader(
        val_set,
        batch_size=config.test_batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )
    if config.use_val_set
    else None
)
test_data_loader = None





loss_func = (torch.nn.CrossEntropyLoss()).to(config.device)
model = (resnet18(num_classes=6) if config.train_discriminator else resnet18()).to(
    config.device
)
optimizer = SGD(model.parameters(), lr=config.lr, weight_decay=0.0005, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, config.num_epochs)

best_val_accuracy = val_acc = 0.0


for epoch in tqdm(range(1, config.num_epochs + 1), desc="Epoch"):
    loss_list = []
    model.train()
    train_loss = train_acc = 0.0
    # Model training
    for (img, labels) in tqdm(train_data_loader, leave=False, desc="Batch"):
        img, labels = img.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        predictions = model(img)
        loss = loss_func(predictions, labels)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        with torch.no_grad():
            correct = torch.argmax(predictions.data, 1) == labels
        train_loss += loss
        train_acc += correct.sum()
    train_loss /= len(train_data_loader.dataset)
    train_acc /= len(train_data_loader.dataset)
    metrics = {"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc}
    scheduler.step()

    def eval_data_loader(loader):
        model.eval()
        loss, acc = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(
                tqdm(loader, desc="Validation: ", leave=False)
            ):
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                predictions = model.forward(inputs)
                loss += loss_func(input=predictions, target=labels).item()
                acc += (torch.argmax(predictions, dim=1) == labels).sum().item()
        loss /= len(loader.dataset)
        acc /= len(loader.dataset)
        return loss, acc

    # Model evaluation on dev set
    if (
        config.use_val_set
        and epoch >= config.validate_start
        and epoch % config.validate_every == 0
    ):
        val_loss, val_acc = eval_data_loader(val_data_loader)
        metrics |= {"val/loss": val_loss, "val/acc": val_acc}
        if val_acc > best_val_accuracy:
            wandb.run.summary["best_val_accuracy"] = val_acc
            if config.save_model:
                torch.save(model.state_dict(), config.save_path + "discriminator.pth")
        print(f"Epoch {epoch}: Val acc: {val_acc:.6f}")

    # if (
    #     not config.train_discriminator
    #     and epoch >= config.validate_start
    #     and epoch % config.validate_every == 0
    # ):
    #     test_loss, test_acc = eval_data_loader(test_data_loader)
    #     metrics |= {"test/loss": test_loss, "test/acc": test_acc}
    #     if val_acc > best_val_accuracy:
    #         wandb.run.summary["best_test_accuracy"] = test_acc

    best_val_accuracy = max(val_acc, best_val_accuracy)

    if config.wandb:
        wandb.log(metrics, step=epoch)
