from ast import parse
import os
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import itertools
import datetime
import argparse
from ml_collections import ConfigDict
import torch

# 6 objects
OBJECTS = [
    "sports car",
    "motorcycle",
    "truck",
    "tractor",
    "double decker bus",
    "bicycle",
]

object_save_labels = [
    "car",
    "motorcycle",
    "truck",
    "tractor",
    "bus",
    "bicycle",
]

# 4 locations
LOCATIONS = [
    "in a grassy field",
    "in a city",
    "in a desert",
    "on a snow-covered field",
]

location_save_labels = [
    "grass",
    "city",
    "desert",
    "snow",
]

# 2 times of day
TIMES_OF_DAY = [
    "during the day",
    "at night",
]

time_save_labels = [
    "day",
    "night",
]

prompt_combinations = list(itertools.product(OBJECTS, LOCATIONS, TIMES_OF_DAY))

prompt_combinations_save_labels = list(
    itertools.product(object_save_labels, location_save_labels, time_save_labels)
)

# 6 objects * 4 locations * 2 times_of_day
# = 48 combinations


def get_prompt(idx):

    object = prompt_combinations[idx][0]
    location = prompt_combinations[idx][1]
    time_of_day = prompt_combinations[idx][2]

    object_save_label = prompt_combinations_save_labels[idx][0]
    location_save_label = prompt_combinations_save_labels[idx][1]
    time_save_label = prompt_combinations_save_labels[idx][2]

    an = "an" if object[0] in "aeiou" else "a"
    body = f" {object} {location}, {time_of_day}, highly detailed, with cinematic lighting, 4k resolution, beautiful composition, hyperrealistic"
    prompt = an + body

    # save_label = f"{time_save_label}/{location_save_label}/{object_save_label}"
    save_label = f"{object_save_label}-{location_save_label}-{time_save_label}"

    return prompt, save_label


# Obtain arguments for array and number of batches to generate
parser = argparse.ArgumentParser()
parser.add_argument("--n_batches", type=int, default=75)
config = ConfigDict(vars(parser.parse_args()))

# Make a directory for each array in Myriad
date_and_time = datetime.datetime.now()
exp_name = date_and_time.strftime("%d-%b__%H-%M")
path = f"./gen_images/{exp_name}"
os.makedirs(path, exist_ok=True)

# The model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", use_auth_token=True
)
pipe = pipe.to("cuda")
# generate images n_batches * 48
with tqdm(total=config.n_batches * 48, leave=False) as pbar:
    for batch_idx in range(config.n_batches):
        comb_idx = 0
        sub_dir = f"{path}/{batch_idx}"
        os.makedirs(sub_dir, exist_ok=True)

        # 48 combinations
        while comb_idx < 48:

            # Obtain 3 samples
            prompt, save_label = get_prompt(comb_idx)

            prompt_list = [prompt, prompt, prompt]
            # want to get three samples out, and if theres a nsfw one, generate until we have three

            output = pipe(prompt_list)

            samples, nsfw = output["sample"], output["nsfw_content_detected"]
            if sum(nsfw) > 0:
                continue
            for idx, sample in enumerate(samples):
                sample.save(f"{sub_dir}/{(comb_idx + 1) * (idx + 1)}_{save_label}.png")
            comb_idx += 1
            pbar.update(1)
