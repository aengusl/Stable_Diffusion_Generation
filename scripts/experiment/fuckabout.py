from ast import parse
import os
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import itertools
import datetime
import argparse
from ml_collections import ConfigDict
import torch
import os
from sre_parse import CATEGORIES
import torch
import pandas as pd
from skimage import io #transform
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
import math as m

root = 'gen_images/sorted/object_balanced_30-Sep__13-10_100x100_fromsize-29313'

data = datasets.ImageFolder(
            root=root, transform=transforms.transforms.ToTensor()
        )
loader = DataLoader(data, batch_size=8)
count = 0
for (imgs, labels) in loader:
    print(imgs.shape)
    prev_imgs = imgs
    

    if count > 1:
        break
    if count > 0:
        group = torch.cat((prev_imgs, imgs), 0)
        print(group.shape)
    count+= 1
print('group')
print(group[0:8,:,:,:].shape)

# directory = 'gen_images/unsorted/100x100__29-Sep__21-41-40'

# print(os.listdir(directory)[0])

# object_labels = ["car", "motorcycle", "truck", "tractor", "bus", "bicycle"]
# location_labels = ["grass", "city", "desert", "snow"]
# time_labels = ["day", "night"]

# all_causal_combinations = list(itertools.product(location_labels, time_labels))
# counter = 0
# for i in range(8):
#     train_comb = all_causal_combinations.pop(i)
#     for j in range(7):
#         test_comb = all_causal_combinations.pop(j)
#         if train_comb[0] != test_comb[0]:
#             counter += 1
#         all_causal_combinations = list(itertools.product(location_labels, time_labels))
#         if (counter + 1) % 8 == 0:
#             print(train_comb)
#             print(test_comb)
#             print('')


# # 3. one is day, one is night
# day_comb = []
# night_comb = []
# for i in range(4):
#     day_comb.append(all_causal_combinations[2*i])
#     night_comb.append(all_causal_combinations[2*i + 1])

# print(day_comb)

# for i in tqdm(range(2)):
#     if i == 0:
#         train_comb = day_comb
#         test_comb = night_comb
#         print(train_comb)
#     else:
#         train_comb = night_comb
#         test_comb = day_comb
#         print(train_comb)

#     # Make lists out of tuples
#     train_locations = []
#     for i in range(len(train_comb)):
#         for location in location_labels:
#             if location in train_comb[i][0] and location not in train_locations:
#                 train_locations.append(location)

#     train_times = []
#     for i in range(len(train_comb)):
#         for time in time_labels:
#             if time in train_comb[i][1] and time not in train_times:
#                 train_times.append(time)

#     test_locations = []
#     for i in range(len(test_comb)):
#         for location in location_labels:
#             if location in test_comb[i][0] and location not in test_locations:
#                 test_locations.append(location)

#     test_times = []
#     for i in range(len(test_comb)):
#         for time in time_labels:
#             if time in test_comb[i][1] and time not in test_times:
#                 test_times.append(time)

#     val_locations  = train_locations
#     val_times = train_times

#     # Make the causal combination
#     causal_combination = {}
#     causal_combination['train_locations'] = train_locations
#     causal_combination['train_times'] = train_times
#     causal_combination['val_locations'] = val_locations
#     causal_combination['val_times'] = val_times
#     causal_combination['test_locations'] = test_locations
#     causal_combination['test_times'] = test_times



# # 5. 6 in train (day/night), 2 in test
# for i in range(4):
#     train_comb = []
#     test_comb = []
#     test_comb.append(all_causal_combinations.pop(2*i))
#     test_comb.append(all_causal_combinations.pop(2*i))
#     train_comb = all_causal_combinations
#     all_causal_combinations = list(itertools.product(location_labels, time_labels))

# print(train_comb)

# train_locations = []
# for i in range(len(train_comb)):
#     for location in location_labels:
#         if location in train_comb[i][0] and location not in train_locations:
#             train_locations.append(location)
# print(train_locations)



# # 4. 4 in train (day/night), 4 in test(day/night)
# for i in range(3):
#     for j in range(i,3):
#         train_comb = []
#         test_comb = []
#         train_comb.append(all_causal_combinations.pop(2*i))
#         train_comb.append(all_causal_combinations.pop(2*i))
#         train_comb.append(all_causal_combinations.pop(2*j))
#         train_comb.append(all_causal_combinations.pop(2*j))
#         test_commb = all_causal_combinations
#         all_causal_combinations = list(itertools.product(location_labels, time_labels))

# # 3. one is day, one is night
# day_comb = []
# night_comb = []
# combo = []
# for i in range(4):
#     day_comb = all_causal_combinations[2*i]
#     night_comb = all_causal_combinations[2*i + 1]

# for i in range(2):
#     if i == 0:
#         train_comb = day_comb
#         test_comb = night_comb
#     else:
#         train_comb = night_comb
#         test_comb = day_comb

#     combo.append(day_comb)
#     combo.append(night_comb)
# causal_comb_ = combo
# print(combo)
# for _ in range(2):
#     train_comb = causal_comb_.pop(i)
#     test_comb = causal_comb_[0]
#     causal_comb_ = combo
# print(train_comb)
# print(test_comb)


# for i in range(4):
#     for j in range(3):
    
   
#         train_comb1 = all_causal_combinations.pop(2*i)
#         train_comb2 = all_causal_combinations.pop(2*i)

#         test_comb1 = all_causal_combinations.pop(2*j)
#         test_comb2 = all_causal_combinations.pop(2*j)

#         all_causal_combinations = list(itertools.product(location_labels, time_labels))





# print(object_labels)
# string = ''
# for object in object_labels:
#     string += object + '-' 
# string = string[:-1]
# print(string)


# for j in range(10):
#     for i in range(10):
#         print(f"j is {j}, i is {i}")
#         if i == 5:
#             break

# def intersection(lst1, lst2):
#     lst3 = [value for value in lst1 if value in lst2]
#     return lst3

# # use mean and std from CIFAR100
# mean=[0.4914, 0.4822, 0.4465]
# std=[0.2023, 0.1994, 0.2010]

# # build test and validation transforms
# test_transforms = transforms.transforms.Compose([
#                     transforms.transforms.ToTensor(),
#                     transforms.transforms.Normalize(mean, std),
#                                             ])
# root_dir = "gen_images/32x32_27-Sep__09-48/"
# train_times=['day', 'night']
# train_locations=['city', 'grass', 'snow','desert']
# train_data_list = []
# val_data_list = []

# for time in train_times:
#         for location in train_locations:
            
#             path = root_dir + f"{time}/{location}/"
#             train_data_ = datasets.ImageFolder(root=path, transform=test_transforms)
#             val_data_ = datasets.ImageFolder(root=path, transform=test_transforms)

#             train_data_list.append(train_data_)
#             val_data_list.append(val_data_)

# train_data = ConcatDataset(train_data_list)
# val_data = ConcatDataset(val_data_list)

# n = len(train_data)
# train_idx = random.sample(range(n), m.floor(0.8 * n))
# val_idx = []
# for i in range(n):
#     if i not in train_idx:
#         val_idx.append(i)

# print(intersection(train_idx, val_idx))

# train_data_lens = [len(dataset) for dataset in train_data_list]
# print(train_data_lens)
# train_min_len = min(train_data_lens)
# for dataset in train_data_list:
#     if len(dataset) > train_min_len:
#         dataset = dataset[0:train_min_len]


# train_data = ConcatDataset(train_data_list)


# counter = [0 for object in object_labels]
# for i in range(len(train_data)):
#     _, label = train_data[i]
#     counter[label] += 1



# list1 = ['duck', 'goose']
# list2 = ['apple', 'orange']
# list3 = ['apple', 'pear']

# prod1 = list(itertools.product(
#     list1, list2
# ))

# prod2 = list(itertools.product(
#     list1, list3
# ))

# prod3 = list(itertools.product(
#     list2, list3
# ))


# if len(intersection(prod1, prod2)) > 0:
#     print('yup1')
# if len(intersection(prod1, prod3)) > 0:
#     print('yup2')
# if len(intersection(prod3, prod2)) > 0:
#     print('yup3')

