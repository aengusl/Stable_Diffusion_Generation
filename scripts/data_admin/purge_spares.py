import os
from tqdm import tqdm
from PIL import Image
from datetime import datetime

"""
There are spare files in desert folder, so any images with count > 900 
will be removed
"""

root = '/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/all-iters_100_images/desert'
for folder in os.listdir(root):
    class_folder = os.path.join(root, folder)
    for file in os.listdir(class_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            count_png = filename.split('_')[-1]
            count = int(count_png.split('.')[0])
            if count > 900:
                file_path = os.path.join(class_folder, file)
                os.remove(file_path)

            


















