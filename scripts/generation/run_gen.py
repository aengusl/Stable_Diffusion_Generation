import os
from datetime import datetime

import torch
from torch import autocast
from torchvision import datasets
from tqdm import tqdm

from assets import DATASET_TO_LABELS
from config_gen import get_config
from pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline, preprocess

config = get_config()

DATASETS = {
    "svhn": datasets.SVHN,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "imagenet": datasets.ImageNet,
}


train_dataset = DATASETS[config.dataset.lower()](
    root=config.dataset_path,
    train=True,
    download=True,
)

(img, label) = train_dataset[0]


# load the pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
).to(config.device)

EXP_NAME = f"data={config.dataset}_seed={config.seed}"
config.generated_images_path += EXP_NAME + "/{:%Y_%m_%d_%H_%M_%S_%f}/".format(
    datetime.now()
)
label_descriptions = DATASET_TO_LABELS[config.dataset.lower()]
os.makedirs(config.generated_images_path, exist_ok=True)
for class_idx in list(label_descriptions.keys()):
    os.makedirs(config.generated_images_path + str(class_idx), exist_ok=True)

counter = idx = config.start_idx
pbar = tqdm(total=config.end_idx - config.start_idx)

while idx < config.end_idx:
    img_tensor_batch, img_pil_batch, labels_batch = [], [], []
    for _ in range(config.diffusion_batch_size):
        img, label = train_dataset[idx]
        img = img.resize((512, 512))
        img_pil_batch.append(img)
        tensor_img = preprocess(img)
        img_tensor_batch.append(tensor_img)
        labels_batch.append(label)
        idx += 1
    prompts_batch = [
        label_descriptions[label] + " realistic 4k detailed" for label in labels_batch
    ]
    img_tensor_batch = torch.vstack(img_tensor_batch)
    with autocast(config.device):
        fake_imgs = pipe(
            prompt=prompts_batch,
            init_image=img_tensor_batch,
            strength=0.5,
            guidance_scale=7.5,
        )["sample"]
        for fake_img, real_img, label, prompt in zip(
            fake_imgs, img_pil_batch, labels_batch, prompts_batch
        ):
            fake_img.save(f"{config.generated_images_path}{label}/fake_{counter}.png")
            real_img.save(f"{config.generated_images_path}{label}/real_{counter}.png")
            counter += 1
    pbar.update(config.diffusion_batch_size)
