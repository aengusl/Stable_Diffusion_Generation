
import os
from tqdm import tqdm
from PIL import Image
from datetime import datetime


"""
This code takes in an object balanced dataset, and creates a new dataset with object imbalances
which depend on the location

eg.
- cities: no tractors
- grass, snow: no bicycles, cars, buses
- desert has everything
"""

# All possible causal factors
object_labels = ["car", "motorcycle", "truck", "tractor", "bus", "bicycle"]
location_labels = ["grass", "city", "desert", "snow"]
time_labels = ["day", "night"]

# Make new object imbalanced folders
EXP_NAME = "{:%d-%b_%H-%M}".format(datetime.now())
balanced_root = "gen_images/100_balanced_07-Nov_15-11"
imbalanced_root = f"gen_images/100_imbalanced_{EXP_NAME}"
for time in time_labels:
    for location in location_labels:
        for object in object_labels:
            path = os.path.join(imbalanced_root, f"{time}/{location}/{object}")
            os.makedirs(path, exist_ok=True)

# Decide the imbalances
city_objects = ["car", "motorcycle", "truck", "bus", "bicycle"]
grass_objects = ["motorcycle", "truck", "tractor"]
snow_objects = grass_objects
desert_objects = ["car", "motorcycle", "truck", "tractor", "bus", "bicycle"]

dict = {}
dict['grass'] = grass_objects
dict['snow'] = snow_objects
dict['city'] = city_objects
dict['desert'] = desert_objects

# Create new folder
for time in tqdm(time_labels):
    for location in tqdm(location_labels):
        object_list = dict[location]
        for object in object_list:
            sub_folder = os.path.join(balanced_root, f"{time}/{location}/{object}")
            for file in os.listdir(sub_folder):
                filename = os.fsdecode(file)
                if filename.endswith(".png"):
                    image_path = os.path.join(sub_folder, filename)
                    img = Image.open(image_path)

                    save_folder = os.path.join(imbalanced_root, f"{time}/{location}/{object}")
                    save_path = os.path.join(save_folder, filename)
                    img.save(save_path)



