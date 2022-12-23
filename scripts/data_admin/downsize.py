import os
from tqdm import tqdm
from PIL import Image
from datetime import datetime

"""

Get 512x512 data

- downsize to new dim
- put the data into sorted folder

^ do this for iter-0 data only, then do it for all data

"""

animals = [
    'dragon',
    'labrador',
    'corgi',
    'bulldog',
    'dachsund',
    'owl',
    'eagle',
    'penguin',
    'goose',
    'cat',
    'lion',
    'hamster',
    'stallion',
    'unicorn',
]

locations = [
    'desert',
    'snow',
    'grass',
    'water',
    'cage',
    'jungle',
]

# Establish roots, and create target directories
root_512_images = "/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/15-Dec_16-23-21_bigboy/iter-0"
new_dim = 100
exp_time = "{:%d-%b_%H-%M}".format(datetime.now())
root_100_images = f"/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/iter-0_100_images"
os.makedirs(root_small_image_sorted, exist_ok=True)


# make directories for sorted folders
for time in time_save_labels:
    for location in location_save_labels:
        for object in object_save_labels:
            path = os.path.join(root_small_image_sorted, f"{time}/{location}/{object}")
            os.makedirs(path, exist_ok=True)

# Now append the files into the sorted and unsorted folders
for file in tqdm(os.listdir(directory_of_big_images)): 
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        image_path = os.path.join(directory_of_big_images, filename)
        big_image = Image.open(image_path)

        # Process the filename
        filesplit = filename.split("_")
        counter = int(filesplit[0])
        labels = filesplit[-1]
        labels = labels.split(".")[0]
        labels = labels.split("-")

        # get causal information
        object = labels[0]
        location = labels[1]
        time = labels[2]

        # get names
        unsort_name = f'{counter}_{filesplit[-1]}'
        sort_name = f'{time}/{location}/{object}/{counter}_image.png'

        # downsize
        small_image = big_image.resize((new_dim, new_dim), Image.ANTIALIAS)

        # get paths
        small_image_unsorted = os.path.join(root_small_image_unsorted, unsort_name)
        small_image_sorted = os.path.join(root_small_image_sorted, sort_name)

        # start saving 
        small_image.save(small_image_unsorted)
        small_image.save(small_image_sorted)


# # Count the number of files in the big image directory
# num_files_in_big_image_direct = 0
# for file in tqdm(os.listdir(directory_of_big_images)):
#     filename = os.fsdecode(file)
#     if filename.endswith(".png"):
#         labels = filename.split("_")
#         counter_ = int(labels[0])
#         print(labels[0])
#         if counter_ > num_files_in_big_image_direct:
#             num_files_in_big_image_direct = counter_

# print(f'num files in big image directory is {num_files_in_big_image_direct}')