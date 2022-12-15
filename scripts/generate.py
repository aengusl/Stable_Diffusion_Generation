import os
import random
from datetime import datetime

from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from assets import DATASET_TO_LABELS, PROMPTS
from config_gen import get_config
from causal_prompts import get_prompt

# Get config information
config = get_config()

# Define experiment folder
exp_name = "/{:%a-%d-%m_%H-%M-%S}/".format(datetime.now())
config.generated_images_path += exp_name

# Get model
pipe = StableDiffusionPipeline.from_pretrained(config.model_id, use_auth_token=True)
pipe = pipe.to(config.device)

# Make a directory of where to save images
os.makedirs(config.generated_images_path, exist_ok=True)

# Generate images
for i in range(config.n_images):
    
    # Obtain sample
    prompt = get_prompt()
    image = pipe(config.prompt).image[0]

    # Saving the image
    str = prompt[:len(prompt)-42]
    image.save(f"{config.generated_images_path}str.png")
