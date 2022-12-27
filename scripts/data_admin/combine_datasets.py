import os
from tqdm import tqdm
from PIL import Image
from datetime import datetime

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

datasets = []
datasets.append('/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/all-iters_100_images_bigboy')
datasets.append('/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/all-iters_100_images_jean')

# Get root for the destination and make directories
root_combined = '/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/combined_100_images'
for loc in locations:
    for anim in animals:
        path = os.path.join(root_combined, f"{loc}/{anim}")
        os.makedirs(path, exist_ok=True)

# Retrieve images from each dataset and append them to the list
for i in tqdm(range(len(datasets))):
    root_dataset = datasets[i]
    for loc in tqdm(locations):
        for anim in animals:
            print(f'saving {loc}, {anim}, dataset {root_dataset.split("/")[-1]}')
            count = 900*i + 1
            path_dataset = os.path.join(root_dataset, f"{loc}/{anim}")
            path_combined = os.path.join(root_combined, f"{loc}/{anim}")
            for file in os.listdir(path_dataset):
                filename = os.fsdecode(file)
                if filename.endswith(".png"):
                    image_dataset_path = os.path.join(path_dataset, filename)
                    image_dataset = Image.open(image_dataset_path)
                    image_combined_name = f"image_{count}.png"
                    save_combined_path = os.path.join(path_combined, image_combined_name)
                    image_dataset.save(save_combined_path)
                    count += 1