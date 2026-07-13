import datetime
import os
import random

import numpy as np
import torch
import yaml


def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file (e.g. configs/default.yaml).

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r", encoding="utf-8") as stream:
        cfgs = yaml.safe_load(stream)
    return cfgs


def load_yaml(config_path):
    """Load YAML configuration from configs/ or an explicit path.

    Args:
        config_path (str): Config filename or path. Bare filenames are resolved
            under ``configs/`` (e.g. ``default.yaml`` -> ``configs/default.yaml``).

    Returns:
        dict: Parsed configuration.
    """
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        config_path = os.path.join("configs", config_path)
    return load_config(config_path)


def get_experiment_dir(cfgs):
    """Return the experiment output directory.

    Args:
        cfgs (dict): Configuration dictionary.

    Returns:
        str: ``output_dir/exp_name``.
    """
    return os.path.join(cfgs["output_dir"], cfgs["exp_name"])


def save_yaml(cfgs, save_path=None):
    """Save resolved configuration to YAML.

    Args:
        cfgs (dict): Configuration dictionary.
        save_path (str, optional): Destination file path. Defaults to
            ``<output_dir>/<exp_name>/config.yaml``.
    """
    if save_path is None:
        save_path = os.path.join(get_experiment_dir(cfgs), "config.yaml")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fp:
        yaml.dump(cfgs, fp, default_flow_style=False)


def fix_seed(seed):
    """Fix random seeds for reproducibility.

    Args:
        seed (int): Seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Initialize random seeds for DataLoader workers.

    See:
        https://pytorch.org/docs/master/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_expname(name):
    """Build an experiment directory name with a date prefix.

    Args:
        name (str): User-provided experiment suffix.

    Returns:
        str: Date-prefixed experiment name.
    """
    return str(datetime.date.today()) + "_" + name


def prepare_exp_dir(cfgs):
    """Prepare experiment output directories.

    Args:
        cfgs (dict): Configuration dictionary.
    """
    experiment_dir = get_experiment_dir(cfgs)
    save_dirs = [
        os.path.join(experiment_dir, "checkpoints"),
        os.path.join(experiment_dir, "tensorboard"),
        os.path.join(experiment_dir, "results"),
    ]

    for save_dir in save_dirs:
        os.makedirs(save_dir, exist_ok=True)


def load_checkpoint(checkpoint_path, device):
    """Load a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device: Target device for deserialization.

    Returns:
        dict: Checkpoint state.
    """
    print("Loading {} checkpoint model".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def save_checkpoint(cfgs, model, optimizer, epoch):
    """Save the best checkpoint file.

    Args:
        cfgs (dict): Configuration dictionary.
        model: Model to save.
        optimizer: Optimizer to save.
        epoch (int): Current epoch number.
    """
    save_path = os.path.join(get_experiment_dir(cfgs), "checkpoints", "best.pth.tar")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        save_path,
    )


def select_optimizer(model, cfgs):
    """Choose an optimizer from the training configuration.

    Args:
        model: Model whose parameters will be optimized.
        cfgs (dict): Configuration dictionary.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.
    """
    optimizer_name = cfgs["train"]["optimizer"]
    lr = cfgs["train"]["lr"]
    weight_decay = cfgs["train"]["weight_decay"]

    if optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "nag":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.7, nesterov=True
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    elif optimizer_name == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} [{avg" + self.fmt + "}]"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Display training progress with metric meters."""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
