import datetime
from diffusers import StableDiffusionPipeline
import torch
import argparse
from ml_collections import ConfigDict
import os

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="a stand up comedian in the style of van gogh")
config = ConfigDict(vars(parser.parse_args()))

date_and_time = datetime.datetime.now()
exp_name = date_and_time.strftime("%H-%M__%d-%b__one_at_a_time")
path = f"./gen_images/"
os.makedirs(path, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", 
                use_auth_token=True, 
                )
pipe = pipe.to("cuda")

image = pipe(config.prompt).image[0]
image.save(f"{path}{exp_name}")



