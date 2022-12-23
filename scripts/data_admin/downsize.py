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

exp_time = "{:%d-%b_%H-%M}".format(datetime.now())

animals = [
    'fire-breathing',
    'labrador',
    'welsh',
    'bulldog',
    'dachsund',
    'owl',
    'bald',
    'emperor',
    'goose',
    'house',
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

# Get root for the destination and make directories
root_100_images = '/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/all-iters_100_images'
for loc in locations:
    for anim in animals:
        path = os.path.join(root_100_images, f"{loc}/{anim}")
        os.makedirs(path, exist_ok=True)

# For loop where we iterate over the iter-i datasets
for i in tqdm(range(4)):
    root_512_images = f'/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/15-Dec_16-23-21_bigboy/iter-{i}'
    for loc in tqdm(locations):
        for anim in animals:
            print(f'saving {loc}, {anim}, iteration {i}')
            # get count to respect the iteration
            count = 1 + 300*i
            path_512 = os.path.join(root_512_images, f"{loc}/{anim}")
            path_100 = os.path.join(root_100_images, f"{loc}/{anim}")
            for file in os.listdir(path_512):
                filename = os.fsdecode(file)
                if filename.endswith(".png"):
                    image_512_path = os.path.join(path_512, filename)
                    image_512 = Image.open(image_512_path)
                    image_100 = image_512.resize((100, 100), Image.Resampling.LANCZOS)
                    image_100_name = f"image_{count}.png"
                    save_100_path = os.path.join(path_100, image_100_name)
                    image_100.save(save_100_path)
                    count += 1

# # Establish roots, and create target directories
# root_512_images = "/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/15-Dec_16-23-21_bigboy/iter-0"
# root_100_images = f"/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/iter-0_100_images"
# os.makedirs(root_100_images, exist_ok=True)

# # make directories for sorted folders
# for loc in locations:
#     for anim in animals:
#         path = os.path.join(root_100_images, f"{loc}/{anim}")
#         os.makedirs(path, exist_ok=True)

# # Downsize the images and save in the right folders
# for loc in tqdm(locations):
#     for anim in animals:
#         print(f'saving {loc}, {anim}')
#         count = 1
#         path_512 = os.path.join(root_512_images, f"{loc}/{anim}")
#         path_100 = os.path.join(root_100_images, f"{loc}/{anim}")
#         for file in os.listdir(path_512):
#             filename = os.fsdecode(file)
#             if filename.endswith(".png"):
#                 image_512_path = os.path.join(path_512, filename)
#                 image_512 = Image.open(image_512_path)
#                 image_100 = image_512.resize((100, 100), Image.Resampling.LANCZOS)
#                 image_100_name = f"image_{count}.png"
#                 save_100_path = os.path.join(path_100, image_100_name)
#                 image_100.save(save_100_path)
#                 count += 1


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



# # Now append the files into the sorted and unsorted folders
# for file in tqdm(os.listdir(root_512_images)): 
#     filename = os.fsdecode(file)
#     if filename.endswith(".png"):
#         image_path = os.path.join(directory_of_big_images, filename)
#         big_image = Image.open(image_path)

#         # Process the filename
#         filesplit = filename.split("_")
#         counter = int(filesplit[0])
#         labels = filesplit[-1]
#         labels = labels.split(".")[0]
#         labels = labels.split("-")

#         # get causal information
#         object = labels[0]
#         location = labels[1]
#         time = labels[2]

#         # get names
#         unsort_name = f'{counter}_{filesplit[-1]}'
#         sort_name = f'{time}/{location}/{object}/{counter}_image.png'

#         # downsize
#         small_image = big_image.resize((new_dim, new_dim), Image.ANTIALIAS)

#         # get paths
#         small_image_unsorted = os.path.join(root_small_image_unsorted, unsort_name)
#         small_image_sorted = os.path.join(root_small_image_sorted, sort_name)

#         # start saving 
#         small_image.save(small_image_unsorted)
#         small_image.save(small_image_sorted)