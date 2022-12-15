import os
from datetime import datetime
from fnmatch import fnmatch
from PIL import Image
from tqdm import tqdm
import csv
import random


"""
Get 100 random samples from each folder
"""


object_save_labels = [
    "car",
    "motorcycle",
    "truck",
    "tractor",
    "bus",
    "bicycle",
]

location_save_labels = [
    "grass",
    "city",
    "desert",
    "snow",
]

time_save_labels = [
    "day",
    "night",
]
begin_exp_time_path = "{:%d-%b__%H-%M-%S}".format(datetime.now())
root = "gen_images/sorted/100x100__29-Sep__21-41-40"
samples_root = os.path.join("gen_images/samples/", begin_exp_time_path)

# Count the number of occurences
dict_of_combs = {}
for time in time_save_labels:
    for location in location_save_labels:
        for object in object_save_labels:
            path = os.path.join(root, f"{time}/{location}/{object}")
            count = 0
            folder = os.path.join(samples_root, f"{time}/{location}/{object}")
            os.makedirs(folder, exist_ok=True)
            for file in tqdm(os.listdir(path)):
                filename = os.fsdecode(file)
                if filename.endswith(".png"):
                    count += 1
                    dict_of_combs[(object, location, time)] = count

# Sample 100 randomly
for time in time_save_labels:
    for location in location_save_labels:
        for object in object_save_labels:
            path = os.path.join(root, f"{time}/{location}/{object}")

            count = dict_of_combs[(object, location, time)]
            indices = random.sample(range(count), 100)
            folder = os.path.join(samples_root, f"{time}/{location}/{object}")
            for i, file in enumerate(tqdm(os.listdir(path))):
                if i in indices:
                    image_path = os.path.join(path, file)
                    img = Image.open(image_path)
                    save_path = os.path.join(folder, file)
                    img.save(save_path)
