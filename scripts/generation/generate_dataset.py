from ast import parse
import os
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import itertools
from datetime import datetime
import argparse
from ml_collections import ConfigDict
from config_gen import get_config
import torch
import random
import numpy as np

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

animals = [
    'fire-breathing dragon',
    'labrador',
    'welsh corgi dog',
    'bulldog',
    'dachsund',
    'owl',
    'bald eagle',
    'emperor penguin',
    'goose',
    'house cat',
    'lion',
    'hamster',
    'stallion horse',
    'unicorn',
]

locations = [
    'in a hot, orange sand desert',
    'in a field covered with snow',
    'in a field of grass',
    'deep under water',
    'in a cage',
    'on an empty highway',
    'in a jungle',
]

# Get config parameters
config = get_config()
set_seed(config.seed)
batch_size = config.batch_size
machine_name = config.machine_name
num_iters = config.num_iters
begin_exp_time = "{:%d-%b_%H-%M-%S}".format(datetime.now())

# The model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", use_auth_token=True
)
pipe = pipe.to("cuda")

# Create a new folder for every iteration through the dataset
for iter in range(num_iters):
    # Loop over animals and locations. 
    # batch_size measures how many samples from a given animal-location prompt are created (should be multiple of 3)
    for animal in tqdm(animals):
        for location in locations:
            comb_idx = 0
            # Generate batch_size number of samples from the prompt
            while 3 * comb_idx < batch_size:
                prompt = f"{animal} {location}, highly detailed, with cinematic lighting, 4k resolution, beautiful composition, hyperrealistic"
                anim_split = animal.split(' ')[0]
                loc_split = location.split(' ')[-1]
                save_label = f"/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/{begin_exp_time}_{machine_name}/iter-{iter}/{loc_split}/{anim_split}"
                os.makedirs(save_label, exist_ok=True)
                # Three at a time
                prompt_list = [prompt, prompt, prompt]
                output = pipe(prompt_list)
                samples = output.images
                # Want to keep the number of samples per prompt consistent, so loops with NSFW samples are repeated 
                nsfw_count = output.nsfw_content_detected
                if sum(nsfw_count) > 0:
                    print( 'NSFW SUCCESFULLY AVOIDED SAVING')
                    continue
                # Save the images
                for idx, sample in enumerate(samples):
                    save_num = (3 * comb_idx + idx) * (iter + 1) 
                    sample.save(f"{save_label}/{machine_name}_{save_num}.png")
                comb_idx += 1

                # Progress tracker every 90 samples
                if comb_idx % 30 == 0:
                    print(' ')
                    print(f'Samples generated in batch {anim_split}-{loc_split}: {comb_idx * 3}')
                    print(f'Batch_size: {batch_size}')
                    print('Iteration:', iter)
                    now = datetime.now()
                    print('Time:', now)
                    print(' ')