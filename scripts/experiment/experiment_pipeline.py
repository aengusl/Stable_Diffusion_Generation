import math
import os
from datetime import datetime
from fnmatch import fnmatch
import time
import torch
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import itertools
import wandb
from cifar_resnet import resnet18
from config_classifier import get_config
from utils import set_seed
from get_the_data import get_train_valid_test_loaders, intersection
from model_training import model_training

"""
Iterate systematically over the causal combinations defined in the format (location, time). Within each 
iteration, we perform model selection by training --num_repeats randomly initialised models with identical
hyperparameters

config controls:
- dataset
- dataset_path
- num_epochs
- save_path (for the model results; default = saved_models/systematic_results)
- device
- objective
- wandb
- num_repeats (for model selection)
- lr (learning rate)
- batch_size:
    - train_batch_size
    - val_batch_size
    - test_batch_size
"""





"""
Define the experiment iteration 
"""

def run_iteration_per_causal_combination(
    train_combinations,
    val_combinations,
    test_combinations, 
    begin_exp_time_path,
    root_dir,
    config,
    exp_iter
    ):

    """
    Run a test for a given set of distributions
    Args:
    - train combination (list of tuples)
    - val combination (list of tuples)
    - test combination(list of tuples)
    - root_dir (path to the data)
    - counter

    Note: this assumes 
    - the root dir points to 32x32 images
    - train and validation distributions are the same
    """

    # Create directory for the iteration
    EXP_NAME = os.path.join(f"alg-{config.objective}_data-{config.dataset}_seed-{config.seed}_iter-{exp_iter}")
    dir_path = os.path.join(config.save_path, begin_exp_time_path, EXP_NAME)
    os.makedirs(dir_path, exist_ok=True)

    # Get train, val and test loaders
    train_loader, val_loader, test_loader = get_train_valid_test_loaders(
        train_combinations,
        val_combinations,
        test_combinations,
        root_dir,
        config
    )

    # Run the training
    model_training(
        train_loader,
        val_loader,
        test_loader,
        train_combinations,
        val_combinations,
        test_combinations,
        EXP_NAME,
        dir_path,
        exp_iter,
        config
    )



"""
Run the experiment

Time imbalancing is specified by whether we use the TI or TB experiments. In order to choose between balanced and imbalanced 
objects, we specify the dataset.
"""

config = get_config()
set_seed(config.seed)
root_dir = config.dataset_path
begin_exp_time_path = "{:%d-%b_%H-%M-%S}".format(datetime.now())

# All possible causal factors
object_labels = ["car", "motorcycle", "truck", "tractor", "bus", "bicycle"]
location_labels = ["grass", "city", "desert", "snow"]
time_labels = ["day", "night"]
all_causal_combinations = list(itertools.product(location_labels, time_labels))

# Objectives we train with
#objectives = ['ERM', "gDRO", 'CLOvE', 'IRM']
objectives = ['CLOvE']

# Time balanced experiments
exp1_TB = {}
exp1_TB['train_combinations'] = [('city', 'day'), ('grass', 'day'), ('desert', 'day')] #[('city', 'night'), ('grass', 'day'), ('desert', 'day')]
exp1_TB['test_combinations'] = [('snow', 'day')] #[('snow', 'day'), ('snow', 'night')]

exp2_TB = {}
exp2_TB['train_combinations'] = [('city', 'day'), ('grass', 'day'), ('snow', 'day')] #[('city', 'night'), ('grass', 'day'), ('snow', 'night')]
exp2_TB['test_combinations'] = [('desert', 'day')] #[('desert', 'day'), ('desert', 'night')]

exp3_TB = {}
exp3_TB['train_combinations'] = [('city', 'night'), ('snow', 'night'), ('grass', 'night')] #[('city', 'night'), ('snow', 'day'), ('grass', 'night')]
exp3_TB['test_combinations'] = [('desert', 'night')] #[('desert', 'day'), ('desert', 'night')]

exp4_TB = {}
exp4_TB['train_combinations'] = [('grass', 'day'), ('desert', 'day'), ('snow', 'day')] #[('grass', 'night'), ('desert', 'night'), ('snow', 'day')]
exp4_TB['test_combinations'] = [('city', 'day')] #[('city', 'day'), ('city', 'night')]

TB_experiments = [exp1_TB, exp2_TB, exp3_TB, exp4_TB]

exp1_TI = {}
exp1_TI['train_combinations'] = [('city', 'night'), ('grass', 'day'), ('desert', 'day')]
exp1_TI['test_combinations'] = [('snow', 'day'), ('snow', 'night')]

exp2_TI = {}
exp2_TI['train_combinations'] = [('city', 'night'), ('grass', 'day'), ('snow', 'night')]
exp2_TI['test_combinations'] = [('desert', 'day'), ('desert', 'night')]

exp3_TI = {}
exp3_TI['train_combinations'] = [('city', 'night'), ('snow', 'day'), ('grass', 'night')]
exp3_TI['test_combinations'] = [('desert', 'day'), ('desert', 'night')]

exp4_TI = {}
exp4_TI['train_combinations'] = [('grass', 'night'), ('desert', 'night'), ('snow', 'day')]
exp4_TI['test_combinations'] = [('city', 'day'), ('city', 'night')]

TI_experiments = [exp1_TI, exp2_TI, exp3_TI, exp4_TI]

# Run the experiments
config.dataset = 'OITI'
exp_iter = 0
for exp in tqdm(TB_experiments):
    for objective in tqdm(objectives):

        train_combinations = exp['train_combinations']
        test_combinations = exp['test_combinations']
        val_combinations = train_combinations
        config.objective = objective

        run_iteration_per_causal_combination(
                    train_combinations,
                    val_combinations,
                    test_combinations, 
                    begin_exp_time_path,
                    root_dir,
                    config,
                    exp_iter
                    )
        
        exp_iter += 1

