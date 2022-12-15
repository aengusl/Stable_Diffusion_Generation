import os
import datetime
from fnmatch import fnmatch
from PIL import Image
from tqdm import tqdm
import csv


object_save_labels = ["car", "motorcycle", "truck","tractor","bus","bicycle"]
location_save_labels = ["grass", "city", "desert", "snow"]
time_save_labels = ["day", "night"]

# root = "gen_images/sorted/object_balanced_32x32__27-Sep__20-37-49"

# for time in time_save_labels:
#     for location in location_save_labels:
#         for object in object_save_labels:
#             path = os.path.join(root, f"{time}/{location}/{object}")
#             os.makedirs(path, exist_ok=True)


'''
Run this code for each folder, overnight
'''

new_root_folder = 'gen_images/'
os.makedirs(new_root_folder, exist_ok=True)
# counter = 0


# for dir in tqdm(os.listdir('b1_new/gen_images/28-Sep__10-05')):
#     subfolder = os.path.join('b1_new/gen_images/28-Sep__10-05', dir)
#     for file in os.listdir(subfolder):
#         filename = os.fsdecode(file)
#         if filename.endswith(".png"):
#             counter += 1
#             image_path = os.path.join(subfolder, filename)
#             img = Image.open(image_path)
#             filesplit = filename.split("_")
#             new_name = f'{counter}_{filesplit[-1]}'
#             new_path = os.path.join(new_root_folder, new_name)
#             img.save(new_path)

# for dir in tqdm(os.listdir('b2_new/gen_images/28-Sep__10-05')):
#     subfolder = os.path.join('b2_new/gen_images/28-Sep__10-05', dir)
#     for file in os.listdir(subfolder):
#         filename = os.fsdecode(file)
#         if filename.endswith(".png"):
#             counter += 1
#             image_path = os.path.join(subfolder, filename)
#             img = Image.open(image_path)
#             filesplit = filename.split("_")
#             new_name = f'{counter}_{filesplit[-1]}'
#             new_path = os.path.join(new_root_folder, new_name)
#             img.save(new_path)

counter = 13929
for file in tqdm(os.listdir('batch1/gen_images/24-Sep__21-03__array-1')):
    #subfolder = os.path.join('b1_new', dir)
    # for file in dir:
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        counter += 1
        image_path = os.path.join('batch1/gen_images/24-Sep__21-03__array-1', filename)
        img = Image.open(image_path)
        filesplit = filename.split("_")
        new_name = f'{counter}_{filesplit[-1]}'
        new_path = os.path.join(new_root_folder, new_name)
        img.save(new_path)

for file in tqdm(os.listdir('batch2/gen_images/24-Sep__21-02__array-1')):
    #subfolder = os.path.join('b1_new', dir)
    # for file in dir:
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        counter += 1
        image_path = os.path.join('batch2/gen_images/24-Sep__21-02__array-1', filename)
        img = Image.open(image_path)
        filesplit = filename.split("_")
        new_name = f'{counter}_{filesplit[-1]}'
        new_path = os.path.join(new_root_folder, new_name)
        img.save(new_path)

for dir in tqdm(os.listdir('last_one_1/29-Sep__00-47')):
    subfolder = os.path.join('last_one_1/29-Sep__00-47', dir)
    for file in os.listdir(subfolder):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            counter += 1
            image_path = os.path.join(subfolder, filename)
            img = Image.open(image_path)
            filesplit = filename.split("_")
            new_name = f'{counter}_{filesplit[-1]}'
            new_path = os.path.join(new_root_folder, new_name)
            img.save(new_path)

for dir in tqdm(os.listdir('last_one_2/29-Sep__00-47')):
    subfolder = os.path.join('last_one_2/29-Sep__00-47', dir)
    for file in os.listdir(subfolder):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            counter += 1
            image_path = os.path.join(subfolder, filename)
            img = Image.open(image_path)
            filesplit = filename.split("_")
            new_name = f'{counter}_{filesplit[-1]}'
            new_path = os.path.join(new_root_folder, new_name)
            img.save(new_path)