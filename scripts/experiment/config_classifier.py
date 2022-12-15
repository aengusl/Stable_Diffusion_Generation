from ml_collections import ConfigDict
import argparse


def get_config():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    # Data
    parser.add_argument("--dataset", type=str, default="OBTB")
    parser.add_argument("--dataset_path", type=str, default="gen_images/100_imbalanced_09-Nov_08-56")
    parser.add_argument("--object_imbalancing", type=str2bool, default=False)

    # Logging and Saving
    parser.add_argument("--wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="Clean_Workshop_07-Nov")
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument(
        "--save_path", type=str, default="./saved_models/systematic_results/"
    )
    # Classifier
    parser.add_argument("--classifier_arch", type=str, default="resnet34")
    parser.add_argument("--fake_dataset_path", type=str, default="~/data/fake/")
    parser.add_argument("--use_val_set", type=str2bool, default=True)
    parser.add_argument("--augment_data", type=str2bool, default=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument('--random_crop_size', type=int, default = 80)
    parser.add_argument('--objective', type=str, default = 'ERM')    

    # Optimization
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_repeats", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--validate_every", type=int, default=5)
    parser.add_argument("--validate_start", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    
    # Output
    parser.add_argument("--pretrained_model_path", type=str, default="~/models/")
    config = ConfigDict(vars(parser.parse_args()))

    return config
