"""Evaluation entry point for MAGE-Vein."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import GroupedFingerVeinDataset
from solver import Solver, resolve_device
from utils import fix_seed, load_config, worker_init_fn


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MAGE-Vein.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint file (e.g. results/<exp>/checkpoints/best.pth.tar).",
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
        default=None,
        help="Experiment name for output paths. Inferred from --checkpoint when omitted.",
    )
    return parser.parse_args()


def format_device(device_arg: str) -> str:
    """Convert a CLI device argument to a PyTorch device string."""
    if device_arg.lower() == "cpu":
        return "cpu"
    if device_arg.isdigit():
        return f"cuda:{device_arg}"
    return device_arg


def infer_exp_name(checkpoint_path: str) -> str:
    """Infer experiment name from a checkpoint path.

    Expected layout: ``<output_dir>/<exp_name>/checkpoints/best.pth.tar``.
    """
    checkpoint = Path(checkpoint_path)
    if checkpoint.parent.name == "checkpoints" and checkpoint.parent.parent.name:
        return checkpoint.parent.parent.name
    return "evaluation"


def apply_cli_overrides(cfgs: dict, args: argparse.Namespace) -> dict:
    """Merge CLI overrides into the loaded configuration."""
    cfgs["exp_name"] = args.exp_name or infer_exp_name(args.checkpoint)
    cfgs["device"] = format_device(args.device)
    cfgs["num_workers"] = (
        args.num_workers if args.num_workers is not None else cfgs.get("num_workers", 4)
    )
    cfgs["checkpoint_path"] = None
    return cfgs


def set_cuda_device(device_cfg) -> None:
    """Set the active CUDA device when GPU execution is available."""
    device = resolve_device(device_cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device)


def build_test_dataloader(cfgs: dict) -> DataLoader:
    """Build the test DataLoader."""
    data_cfg = cfgs["data"]
    dataset = GroupedFingerVeinDataset(
        csv_path=data_cfg["test_csv"],
        data_root=data_cfg["data_root"],
        mode="test",
        crop_size=data_cfg["crop_size"],
        images_per_group=data_cfg["images_per_group"],
    )

    return DataLoader(
        dataset,
        batch_size=cfgs["train"]["batch_size"],
        shuffle=False,
        num_workers=cfgs["num_workers"],
        worker_init_fn=worker_init_fn,
    )


def main() -> None:
    """Run evaluation with an explicit checkpoint."""
    args = parse_args()
    cfgs = load_config(args.config)
    cfgs = apply_cli_overrides(cfgs, args)

    fix_seed(cfgs["seed"])
    set_cuda_device(cfgs["device"])

    testloader = build_test_dataloader(cfgs)
    solver = Solver(cfgs)
    solver.load_checkpoint(args.checkpoint)
    solver.test(testloader)


if __name__ == "__main__":
    main()
