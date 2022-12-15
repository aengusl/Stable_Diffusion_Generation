


import os
from tqdm import tqdm
from PIL import Image
from datetime import datetime


"""

Count the number of objects in a sorted folder, and create a new folder with equal numbers of objects
for every location

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

# Make directories for balanced dataset
EXP_NAME = "{:%d-%b_%H-%M}".format(datetime.now())
sorted_root_dir = "gen_images/100_sorted_02-Nov_17-57"
new_root_dir = f"gen_images/100_balanced_{EXP_NAME}"
for time in time_save_labels:
    for location in location_save_labels:
        for object in object_save_labels:
            path = os.path.join(new_root_dir, f"{time}/{location}/{object}")
            os.makedirs(path, exist_ok=True)

# Identify the number of objects in each folder of our dataset
object_counter_list = []
for time in tqdm(time_save_labels):
    for location in tqdm(location_save_labels):
        for object in object_save_labels:

            counter = 0
            sub_folder = os.path.join(sorted_root_dir, f"{time}/{location}/{object}")
            for file in os.listdir(sub_folder):
                filename = os.fsdecode(file)
                if filename.endswith(".png"):
                    counter += 1
            object_counter_list.append(counter)

print('object counter list is', object_counter_list)
print(' ')
print('min number is ', min(object_counter_list))

# Now create a new folder with balanced object categories
min_object_number = min(object_counter_list)
for time in tqdm(time_save_labels):
    for location in tqdm(location_save_labels):
        for object in object_save_labels:

            object_counter = 0
            sub_folder = os.path.join(sorted_root_dir, f"{time}/{location}/{object}")
            # limit number of files to match across objects
    
            for file in os.listdir(sub_folder):
                if object_counter < min_object_number:
                    filename = os.fsdecode(file)
                    if filename.endswith(".png"):

                        # get the image
                        image_path = os.path.join(sub_folder, filename)
                        img = Image.open(image_path)

                        # get counter info
                        count = filename.split("_")[0]

                        # save the image
                        new_image_path = os.path.join(new_root_dir, f"{time}/{location}/{object}/{count}_image.png")
                        img.save(new_image_path)

                        # count
                        object_counter += 1
                    







