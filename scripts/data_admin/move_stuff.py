import os
from tqdm import tqdm
from PIL import Image
from datetime import datetime


"""
Get 512x512 image data from a downloaded tar file and append it to both
- the unsorted 512x512 image folder
- the sorted 512x512 image folder

Then, convert the data into 32x32 data and append it to both
- the unsorted 32x32 image folder
- the sorted 32x32 image folder

Better..
- do all of this for each image

1. Iterate over the new dataset offered, and begin counter
2. For each file in the dataset
    - append file to unsorted and sorted 512x512 image folder
    - create 32x32 file
    - append 32x32 file to unsorted and sorted 512x512 image folder

To test this before we run properly, create dummy folders
-
"""

#Count the number of files in the unsorted folder so far
unsorted_path = "gen_images/unsorted/32x32__27-Sep__19-52-58"
counter = 0
for file in tqdm(os.listdir(unsorted_path)):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        labels = filename.split("_")
        counter_ = int(labels[0])
        if counter_ > counter:
            counter = counter_

print(counter)
directory_of_downloaded_data = "Stable_Diffusion_Generation/gen_images/512_unsorted"
new_dim = 32
# num_subfolders = 26


root_small_image_sorted = f"gen_images/sorted/32x32__27-Sep__20-37-49"
root_small_image_unsorted = f"gen_images/unsorted/32x32__27-Sep__19-52-58"
root_big_image_sorted = "gen_images/sorted/512x512__27-Sep__20-41-01"
root_big_image_unsorted = "gen_images/unsorted/512x512_27-Sep__19-18-26"

# Now append the files into the sorted and unsorted folders
for i in tqdm(range(74)):
    sub_folder = os.path.join(directory_of_downloaded_data, f"{i}")
    for file in os.listdir(sub_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            image_path = os.path.join(sub_folder, filename)
            big_image = Image.open(image_path)
            filesplit = filename.split("_")[-1]
            labels = filesplit.split(".")[0]
            labels = labels.split("-")

            # get causal information
            object = labels[0]
            location = labels[1]
            time = labels[2]

            # get names
            unsort_name = f'{counter}_{filesplit}'
            sort_name = f'{time}/{location}/{object}/{counter}_image.png'

            # downsize
            small_image = big_image.resize((new_dim, new_dim), Image.ANTIALIAS)

            # get paths
            big_image_unsorted = os.path.join(root_big_image_unsorted, unsort_name)
            big_image_sorted = os.path.join(root_big_image_sorted, sort_name)
            small_image_unsorted = os.path.join(root_small_image_unsorted, unsort_name)
            small_image_sorted = os.path.join(root_small_image_sorted, sort_name)

            os.makedirs(root_small_image_unsorted, exist_ok=True)
            sorted_dir_path = os.path.join(root_small_image_sorted, f'{time}/{location}/{object}')
            os.makedirs(sorted_dir_path, exist_ok=True)

            # start saving 
            big_image.save(big_image_unsorted)
            big_image.save(big_image_sorted)
            small_image.save(small_image_unsorted)
            small_image.save(small_image_sorted)

            # count
            counter+=1











            

