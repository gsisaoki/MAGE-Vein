"""Training entry point for MAGE-Vein."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import GroupedFingerVeinDataset
from solver import Solver, resolve_device
from utils import (
    fix_seed,
    get_expname,
    load_config,
    prepare_exp_dir,
    save_yaml,
    worker_init_fn,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train MAGE-Vein.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index (e.g. 0) or 'cpu'.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Experiment name suffix (date prefix is added automatically).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the optimizer learning rate.",
    )
    return parser.parse_args()


def format_device(device_arg: str) -> str:
    """Convert a CLI device argument to a PyTorch device string."""
    if device_arg.lower() == "cpu":
        return "cpu"
    if device_arg.isdigit():
        return f"cuda:{device_arg}"
    return device_arg


def apply_cli_overrides(cfgs: dict, args: argparse.Namespace) -> dict:
    """Merge CLI overrides into the loaded configuration."""
    cfgs["exp_name"] = get_expname(args.exp_name)
    cfgs["device"] = format_device(args.device)
    cfgs["num_workers"] = (
        args.num_workers if args.num_workers is not None else cfgs.get("num_workers", 4)
    )

    if args.epochs is not None:
        cfgs["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfgs["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfgs["train"]["lr"] = args.lr

    return cfgs


def set_cuda_device(device_cfg) -> None:
    """Set the active CUDA device when GPU execution is available."""
    device = resolve_device(device_cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device)


def build_dataloader(cfgs: dict, split: str) -> DataLoader:
    """Build a DataLoader for the requested dataset split."""
    data_cfg = cfgs["data"]
    csv_path = data_cfg[f"{split}_csv"]
    mode = "train" if split == "train" else split

    dataset = GroupedFingerVeinDataset(
        csv_path=csv_path,
        data_root=data_cfg["data_root"],
        mode=mode,
        crop_size=data_cfg["crop_size"],
        images_per_group=data_cfg["images_per_group"],
    )

    return DataLoader(
        dataset,
        batch_size=cfgs["train"]["batch_size"],
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=cfgs["num_workers"],
        worker_init_fn=worker_init_fn,
    )


def main() -> None:
    """Run the training pipeline."""
    args = parse_args()
    cfgs = load_config(args.config)
    cfgs = apply_cli_overrides(cfgs, args)

    fix_seed(cfgs["seed"])
    set_cuda_device(cfgs["device"])
    prepare_exp_dir(cfgs)

    trainloader = build_dataloader(cfgs, "train")
    valloader = build_dataloader(cfgs, "val")

    solver = Solver(cfgs)
    solver.train(trainloader, valloader)
    save_yaml(cfgs)


if __name__ == "__main__":
    main()
