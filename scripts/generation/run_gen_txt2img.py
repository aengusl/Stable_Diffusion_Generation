import os
import random
from datetime import datetime

from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from assets import DATASET_TO_LABELS, PROMPTS
from config_gen import get_config

config = get_config()
EXP_NAME = f"data={config.dataset}_seed={config.seed}"
config.generated_images_path += EXP_NAME + "/{:%Y_%m_%d_%H_%M_%S_%f}/".format(
    datetime.now()
)
os.makedirs(config.generated_images_path, exist_ok=True)
pipe = StableDiffusionPipeline.from_pretrained(config.model_id, use_auth_token=True)
pipe = pipe.to(config.device)

for class_idx in range(config.start_idx, config.end_idx + 1):
    os.makedirs(f"{config.generated_images_path}/{class_idx}/", exist_ok=True)
    print(f"Generate Class {class_idx}...")
    final_images, final_prompts = [], []
    counter = 0
    with tqdm(total=config.samples_per_class, leave=False) as pbar:
        while len(final_images) < config.samples_per_class:
            label = DATASET_TO_LABELS[config.dataset.lower()][class_idx]
            input_prompts = []
            for _ in range(config.diffusion_batch_size):
                rdn_prompt = label
                if config.add_prompt_suffix:
                    rdn_prompt += " " + random.choice(PROMPTS)
                input_prompts.append(rdn_prompt)
            output = pipe(
                input_prompts,
                num_inference_steps=config.num_inference_steps,
                height=512,
                width=512,
            )
            samples, nsfw = output["sample"], output["nsfw_content_detected"]
            for idx, output in enumerate(zip(samples, nsfw)):
                if not output[1]:
                    output[0].save(
                        f"{config.generated_images_path}/{class_idx}/{counter}.png"
                    )
                    with open(
                        f"{config.generated_images_path}/{class_idx}/{counter}.txt",
                        "w",
                    ) as fp:
                        fp.write(input_prompts[idx])
                    final_images += [samples[idx]]
                    final_prompts += [input_prompts[idx]]
                    counter += 1
                    pbar.update(1)
