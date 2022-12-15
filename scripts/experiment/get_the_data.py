import os
from sre_parse import CATEGORIES
import torch
import pandas as pd
from skimage import io  # transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import itertools
import math as m
from PIL import Image
from torchvision import datasets, transforms
import csv
import os
from torch.utils.data import ConcatDataset, DataLoader
import random

object_labels = ["car", "motorcycle", "truck", "tractor", "bus", "bicycle"]
n_objects = len(object_labels)

location_labels = ["grass", "city", "desert", "snow"]
n_locations = len(location_labels)

time_labels = ["day", "night"]
n_times = len(time_labels)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_train_valid_test_loaders(
    train_combinations,
    val_combinations,
    test_combinations,
    root_dir,
    config,
    train_augment=True,
):

    """
    Return train loader, val loader and test loader, given causal combinations for
    train, val and test, and the root directory where the data is stored (in ImageFolder format)
    """
    val_equals_train = True
    # if len(train_combinations) > 1:
    #     if len(intersection(train_combinations, val_combinations)) > 0:
    #         val_equals_train = True
    #         print(" ")
    #         print(
    #             "Since there is overlap in validation and training distributions, they have been assigned identical distributions"
    #         )
    #         print(" ")
    #     else:
    #         val_equals_train = False

    #     if len(intersection(train_combinations, test_combinations)) > 0:
    #         raise Exception("Training and test distributions overlap")

    toy_train_data_list = []
    train_data_list = []
    val_data_list = []
    test_data_list = []

    # Make a trial training dataset in order to extract means and stds,
    # so that we can initialise all the data with accurate normalization
    for comb in train_combinations:
        location = comb[0]
        time = comb[1]

        path = os.path.join(root_dir, f"{time}/{location}/")
        data = datasets.ImageFolder(
            root=path, transform=transforms.transforms.ToTensor()
        )
        toy_train_data_list.append(data)

    # Obtain means and stds for all transforms
    toy_train_data = ConcatDataset(toy_train_data_list)
    train_data_concat = torch.cat([d[0] for d in DataLoader(toy_train_data)])
    mean = train_data_concat.mean(dim=(0, 2, 3))
    std = train_data_concat.std(dim=(0, 2, 3))

    # Build test and validation transforms
    test_transforms = transforms.transforms.Compose(
        [transforms.transforms.ToTensor(), transforms.transforms.Normalize(mean, std)]
    )

    # Build training data transforms
    if train_augment:
        train_transforms = transforms.transforms.Compose(
            [
                transforms.transforms.RandomCrop(size=(config.random_crop_size, config.random_crop_size), padding=4),
                transforms.transforms.RandomHorizontalFlip(),
                transforms.transforms.ToTensor(),
                transforms.transforms.Normalize(mean, std),
            ]
        )
    else:
        train_transforms = test_transforms

    # If validation distribution = train distr., we split up the data 80/20
    if val_equals_train:
        for comb in train_combinations:
            location = comb[0]
            time = comb[1]

            path = os.path.join(root_dir, f"{time}/{location}/")
            train_data_ = datasets.ImageFolder(
                root=path, transform=train_transforms
            )
            val_data_ = datasets.ImageFolder(root=path, transform=test_transforms)

            # val and train datasets are identical at this point
            train_data_list.append(train_data_)
            val_data_list.append(val_data_)

    # Otherwise, we handle val and train seprately
    else:
        # Make train_data_list
        for comb in train_combinations:
            location = comb[0]
            time = comb[1]

            path = os.path.join(root_dir, f"{time}/{location}/")
            data = datasets.ImageFolder(root=path, transform=train_transforms)
            train_data_list.append(data)

        # Make val_data_list
        for comb in val_combinations:
            location = comb[0]
            time = comb[1]

            path = os.path.join(root_dir + f"{time}/{location}/")
            data = datasets.ImageFolder(root=path, transform=test_transforms)
            val_data_list.append(data)

    # Make test_data_list
    for comb in test_combinations:
        location = comb[0]
        time = comb[1]

        path = os.path.join(root_dir, f"{time}/{location}/")
        data = datasets.ImageFolder(root=path, transform=test_transforms)
        test_data_list.append(data)

    # Balance datasets by permissible time and location combinations
    # before concatenation, to mitigate confounding between train and test
    train_data_lens = [len(data) for data in train_data_list]
    train_min_len = min(train_data_lens)
    for data in train_data_list:
        if len(data) > train_min_len:
            data = Subset(data, range(0, train_min_len))

    val_data_lens = [len(data) for data in val_data_list]
    val_min_len = min(val_data_lens)
    for data in val_data_list:
        if len(data) > val_min_len:
            data = Subset(data, range(0, val_min_len))

    test_data_lens = [len(data) for data in test_data_list]
    test_min_len = min(test_data_lens)
    for data in test_data_list:
        if len(data) > test_min_len:
            data = Subset(data, range(0, test_min_len))

    # Concatenate datasets to get initial train, val and test datasets
    train_data = ConcatDataset(train_data_list)
    val_data = ConcatDataset(val_data_list)
    test_data = ConcatDataset(test_data_list)

    # Split up train and val in the case where validation is in-distribution (ID)
    if val_equals_train:

        n = len(train_data)
        train_idx = random.sample(range(n), m.floor(0.8 * n))
        val_idx = []

        for i in range(n):
            if i not in train_idx:
                val_idx.append(i)

        train_data = Subset(train_data, train_idx)
        val_data = Subset(val_data, val_idx)

    
    if config.objective == 'ERM':

        # Get data loaders
        train_loader = DataLoader(
            train_data,
            batch_size=config.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
        )
    
    else: 

        train_loader = []
        for data in train_data_list:
            loader = DataLoader(
            data,
            batch_size=config.train_batch_size // len(train_data_list),
            shuffle=True,
            pin_memory=False,
            drop_last=False,
        )
            train_loader.append(loader)

    val_loader = DataLoader(
        val_data,
        batch_size=config.val_batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config.test_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
