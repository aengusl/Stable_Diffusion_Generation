import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder

from torch.utils.data import ConcatDataset, DataLoader


class TILO():
    def __init__(self, combinations, root_dir, train_augment=True):
        self.classes = ["car","motorcycle","truck","tractor","bus","bicycle"]
        train_combinations, test_combinations = combinations["train"],combinations["test"]

        self.input_shape = (3,100,100)
        self.num_classes = 6

        toy_train_data_list = []
        train_data_list = []
        test_data_list = []

        if isinstance(train_combinations, dict):
            assert set(sum(train_combinations.keys())) == set(self.classes) , "Classes listed do not match the list of classes ['car'','motorcycle','truck','tractor'','bus','bicycle']"
            for classes,comb_list in train_combinations:
                for comb in comb_list:
                    location = comb[0]
                    time = comb[1]

                    path = os.path.join(root_dir, f"{time}/{location}/")
                    data = ImageFolder(
                        root=path, transform=transforms.transforms.ToTensor()
                    )
                    classes_idx = [data.class_to_idx[c] for c in classes]
                    to_keep_idx = [i for i in range(len(data)) if data.imgs[i][1] in classes_idx]

                    subset = Subset(data, to_keep_idx)

                    toy_train_data_list.append(subset) 
        else:
            for comb in train_combinations:
                location = comb[0]
                time = comb[1]

                path = os.path.join(root_dir, f"{time}/{location}/")
                data = ImageFolder(
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
                    # transforms.transforms.RandomCrop(size=(80, 80))
                    transforms.transforms.RandomHorizontalFlip(),
                    transforms.transforms.ToTensor(),
                    transforms.transforms.Normalize(mean, std),
                ]
            )
        else:
            train_transforms = test_transforms

        for comb in train_combinations:
            location = comb[0]
            time = comb[1]

            path = os.path.join(root_dir, f"{time}/{location}/")
            train_data_ = ImageFolder(
                root=path, transform=train_transforms
            )

            train_data_list.append(train_data_)

        # Make test_data_list
        for comb in test_combinations:
            location = comb[0]
            time = comb[1]

            path = os.path.join(root_dir, f"{time}/{location}/")
            data = ImageFolder(root=path, transform=test_transforms)
            test_data_list.append(data)

        # Balance datasets by permissible time and location combinations
        # before concatenation, to mitigate confounding between train and test
        train_data_lens = [len(data) for data in train_data_list]
        train_min_len = min(train_data_lens)
        for data in train_data_list:
            if len(data) > train_min_len:
                data = Subset(data, range(0, train_min_len))

        test_data_lens = [len(data) for data in test_data_list]
        test_min_len = min(test_data_lens)
        for data in test_data_list:
            if len(data) > test_min_len:
                data = Subset(data, range(0, test_min_len))

        # Concatenate test datasets 
        test_data = ConcatDataset(test_data_list)

        ## List of dataset object for each group
        self.train_datasets = train_data_list
        self.test_dataset = test_data

    def get_train_datasets(self, grouped=True):
        return self.train_datasets if not grouped else ConcatDataset(self.train_datasets)
    
    def get_test_datasets(self, grouped=True):
        return self.test_datasets if not grouped else ConcatDataset(self.test_datasets)

    def get_train_loaders(self, batch_size=32, grouped=True):
        if not grouped:
            return DataLoader(
                        self.get_train_datasets(False),
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=False,
                        drop_last=False,
                    )
        
        loaders = []
        data_list = self.get_train_datasets(True)
        for data in data_list:
            loader = DataLoader(
            data,
            batch_size=batch_size // len(data_list),
            shuffle=True,
            pin_memory=False,
            drop_last=False,
        )
        loaders.append(loader)
        return loaders

    def get_test_loaders(self, batch_size=32, grouped=True):
        if not grouped:
            return DataLoader(
                        self.get_test_datasets(False),
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=False,
                        drop_last=False,
                    )
        
        loaders = []
        data_list = self.get_test_datasets(True)
        for data in data_list:
            loader = DataLoader(
            data,
            batch_size=batch_size // len(data_list),
            shuffle=True,
            pin_memory=False,
            drop_last=False,
        )
        loaders.append(loader)
        return loaders


