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
from cifar_resnet import resnet18
from config_classifier import get_config
from utils import set_seed
from get_the_data import get_train_valid_test_loaders, intersection
import csv
from torch.nn import DataParallel

#These functions execute the algos for a single epoch 
from algorithms.ERM import ERM
from algorithms.gDRO import gDRO
from algorithms.IRM import IRM
from algorithms.CLOvE import CLOvE

# All possible causal factors
object_labels = ["car", "motorcycle", "truck", "tractor", "bus", "bicycle"]
location_labels = ["grass", "city", "desert", "snow"]
time_labels = ["day", "night"]




def model_training(
    train_loader,
    val_loader,
    test_loader,
    train_combinations,
    val_combinations,
    test_combinations,
    EXP_NAME,
    dir_path,
    exp_iter,
    config,
):

    """
    Train a selection of models, perform model selection and report
    results the best model's performance on the test set
    """

    if config.wandb:
        import wandb

        wandb.init(
            name=EXP_NAME,
            project=config.wandb_project,
            config=config.to_dict(),
        )
    
    # Define loss function
    loss_func = (torch.nn.CrossEntropyLoss()).to(config.device)

    st = time.time()
    model_val_accuracies = []
    model_train_accuracies = []

    """
    Begin training loop
    """

    for repeat in range(config.num_repeats):

        # Define model and training procedure
        best_val_accuracy_loop = 0
        model_train_accuracies.append(0)
        torch.manual_seed(repeat)
        model = (resnet18(num_classes=6)).to(config.device)
        if config.objective == 'IRM':
            dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(config.device)
        # spread across available GPUs
        model = DataParallel(model)
        model_path = os.path.join(dir_path, f"discriminator_{repeat+1}.pth")
        optimizer = SGD(
            model.parameters(), lr=config.lr, weight_decay=0.0005, momentum=0.9
        )
        scheduler = CosineAnnealingLR(optimizer, config.num_epochs)
        val_acc = 0

        for epoch in tqdm(range(1, config.num_epochs + 1), desc="Epoch"):     
            
            # Model training
            model.train()

            # Pick objective
            if config.objective == 'ERM':
                train_loss, train_acc = ERM(model,train_loader,optimizer,loss_func,epoch,config)   
            elif config.objective == "gDRO":
                train_loss, train_acc = gDRO(model,train_loader,optimizer,loss_func,epoch,config)   
            elif config.objective == 'IRM':
                train_loss, train_acc = IRM(model,train_loader,optimizer,loss_func,epoch,config, dummy_w)
            elif config.objective == 'CLOvE':
                train_loss, train_acc = CLOvE(model,train_loader,optimizer,loss_func,epoch,config)



            # Model validation
            model.eval()
            val_loss = val_acc = 0.0
            for (img, labels) in val_loader:
                img, labels = img.to(config.device), labels.to(config.device)
                with torch.no_grad():
                    predictions = model(img)
                    loss = loss_func(predictions, labels)
                    correct = torch.argmax(predictions.data, 1) == labels
                val_loss += loss
                val_acc += correct.sum()
            val_loss /= len(val_loader.dataset)
            val_acc /= len(val_loader.dataset)

            # Save best performing model on validation set so far
            if val_acc > best_val_accuracy_loop:
                best_val_accuracy_loop = val_acc
                torch.save(model.state_dict(), model_path)
                model_train_accuracies[repeat] = train_acc
            metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
            }
            if config.wandb:
                wandb.log(metrics, step=epoch)
            print(f" Train acc {train_acc:.3f}, Train loss {train_loss:.3f}")
            print(f" Val acc {val_acc:.3f}, Val loss {val_loss:.3f}")
            print(f" Objective {config.objective}, Exp iter {exp_iter}")

            scheduler.step()

        # Save best val accuracy
        model_val_accuracies.append(best_val_accuracy_loop)
        torch.cuda.empty_cache()

    # Get runtime for training across all repeats
    et = time.time()
    elapsed_time = et - st
    print(f"Training completed, runtime: {elapsed_time:.1f} seconds")
    print(" ")

    """
    Report test set results
    """

    # Initialise counters for overall and class-conditional correct predictions
    test_metrics = {}
    correct_pred_overall = 0
    total_pred_overall = 0
    correct_pred_class = {classname: 0 for classname in object_labels}
    total_pred_class = {classname: 0 for classname in object_labels}

    # Model selection based on best val accuracy
    max_value = max(model_val_accuracies)
    best_model_idx = model_val_accuracies.index(max_value)
    PATH = os.path.join(dir_path, f"discriminator_{best_model_idx+1}.pth")
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # Collect data from test set
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # Calculate correct predictions overall and by class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred_overall += 1
                    correct_pred_class[object_labels[label]] += 1
                total_pred_overall += 1
                total_pred_class[object_labels[label]] += 1

    # Print the distributions we have tested
    if len(intersection(train_combinations, val_combinations)) > 0:
        val_combinations = train_combinations

    print(f"Train distributions - {[comb for comb in train_combinations]}")
    print(f"Validation distributions - {[comb for comb in val_combinations]}")
    print(f"Test distributions - {[comb for comb in test_combinations]}")
    print(" ")

    # Print overall test accuracy
    accuracy = 100 * float(correct_pred_overall) / total_pred_overall
    test_metrics["Overall test accuracy"] = accuracy
    print(f"Overall test accuracy: {accuracy:.1f} %")
    print(" ")

    # Print class conditonal accuracy
    print("Class-conditoned test accuracies")
    for classname, correct_count in correct_pred_class.items():
        accuracy = 100 * float(correct_count) / total_pred_class[classname]
        test_metrics[classname] = accuracy
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
    print(" ")

    # Write results to a text file
    txt_path = os.path.join(dir_path, f"model_perfomance.txt")
    with open(txt_path, "w") as f:
        f.write(f"Experiment runtime: {elapsed_time:.1f} seconds")
        f.write(f"\nNumber of models: {config.num_repeats}")
        f.write(f"\nNumber of max epochs: {config.num_epochs}")
        f.write(f"\nObjective: {config.objective}")
        f.write(f"\n")
        f.write(f"\nData: {config.dataset_path}")
        f.write(f"\nSeed: {config.seed}")
        f.write(f"\nTrain distributions - {[comb for comb in train_combinations]}")
        f.write(f"\nValidation distributions - {[comb for comb in val_combinations]}")
        f.write(f"\nTest distributions - {[comb for comb in test_combinations]}")
        f.write(f"\n")
        f.write(f"\nModel stats: ")
        for i in range(config.num_repeats):
            f.write(
                f"\n-Model {i+1} -- Train Acc:{model_train_accuracies[i]:.2f}, Best Val Acc: {model_val_accuracies[i]:.2f}"
            )
        f.write(f"\n")
        f.write(f"\nChoose model {best_model_idx+1}")
        f.write(f"\n")
        overall_test_acc = test_metrics["Overall test accuracy"]
        f.write(f"\nOverall test accuracy: {overall_test_acc:.3f} %")
        for classname, correct_count in correct_pred_class.items():
            accuracy = 100 * float(correct_count) / total_pred_class[classname]
            f.write(f"\nAccuracy for class: {classname:5s} is {accuracy:.3f} %")

    wandb.log(test_metrics)

    wandb.finish()
