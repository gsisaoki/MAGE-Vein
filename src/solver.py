import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import build_model
from utils import (
    AverageMeter,
    ProgressMeter,
    load_checkpoint,
    save_checkpoint,
    select_optimizer,
)


def resolve_device(device_cfg) -> torch.device:
    """Resolve a device from config with CPU fallback when CUDA is unavailable.

    Args:
        device_cfg: Device string (e.g. ``'cuda:0'``), ``torch.device``, or
            ``None``.

    Returns:
        Resolved ``torch.device``.
    """
    if isinstance(device_cfg, torch.device):
        device = device_cfg
    elif device_cfg is None:
        device = torch.device("cpu")
    else:
        device = torch.device(str(device_cfg))

    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


class Solver(object):
    def __init__(self, cfgs):
        self.device = resolve_device(cfgs.get("device"))
        self.net = build_model(cfgs).to(self.device)
        self.results = dict(iter=0, loss=0)
        self.cfgs = cfgs
        self.epochs = cfgs["train"]["epochs"]
        self.gender_loss_weight = cfgs["train"]["gender_loss_weight"]
        self.early_stop_patience = cfgs["train"]["early_stop_patience"]

        experiment_dir = os.path.join(cfgs["output_dir"], cfgs["exp_name"])
        self.writer = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "tensorboard")
        )
        self.savedir = os.path.join(experiment_dir, "checkpoints")
        self.checkpoint_path = cfgs.get("checkpoint_path")

        self.criterion = nn.MSELoss()
        self.cross = nn.CrossEntropyLoss()
        self.min_mae = float("inf")
        self.optimizer = select_optimizer(self.net, cfgs)

        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)

        self.save_path = os.path.join(experiment_dir, "results")
        self.visualize_list = ["train", "val", "test"]
        for visualize_name in self.visualize_list:
            visualize_save_path = os.path.join(self.save_path, visualize_name)
            os.makedirs(visualize_save_path, exist_ok=True)

        self.metric_list = ["loss"]

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model and optimizer state from a checkpoint file.

        Args:
            checkpoint_path: Path to a ``.pth.tar`` checkpoint.
        """
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def train(self, trainloader, valloader):
        print("Start Training...")
        meter_list = [AverageMeter(meter) for meter in self.metric_list]
        progress = ProgressMeter(
            self.cfgs["train"]["epochs"], meter_list, prefix="Training: "
        )
        n_earlystop = 0
        for epoch in range(1, self.epochs + 1):
            results = self.trainer("train", trainloader, epoch)
            for metric, meter in zip(self.metric_list, meter_list):
                meter.update(results[metric] / results["iter"])
            train_loss = results["loss"] / results["iter"]
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            progress.display(epoch)
            for meter in meter_list:
                meter.reset()

            if len(valloader) == 0:
                save_checkpoint(self.cfgs, self.net, self.optimizer, epoch)
            else:
                mae = self.val(epoch, valloader)
                if self.min_mae > mae:
                    n_earlystop = 0
                    self.min_mae = mae
                    save_checkpoint(self.cfgs, self.net, self.optimizer, epoch)
                else:
                    n_earlystop += 1
                if n_earlystop == self.early_stop_patience:
                    break
        print("End Training")

    def val(self, epoch, valloader):
        if len(valloader) == 0:
            return 0

        print("Start Validation...")
        with torch.no_grad():
            results = self.trainer("val", valloader, epoch)
        if results is None:
            print("Error: trainer returned None during validation.")
            return float("inf")

        preds = results["preds"].flatten()
        labels = results["labels"].flatten()
        mae = np.mean(np.abs(preds - labels))

        val_loss = results["loss"] / results["iter"]
        print(f"MSE: {val_loss:.4f} | MAE: {mae:.2f}")
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("MAE/val", mae, epoch)

        return mae

    def test(self, testloader):
        print("Start Evaluation...")
        self.results = dict(loss=0, iter=0)

        with torch.no_grad():
            results = self.trainer("test", testloader, 1)

        preds = results["preds"].reshape(-1)
        labels = results["labels"].reshape(-1)

        errors = preds - labels
        abs_errors = np.abs(errors)
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))

        std_error = np.std(errors)
        cs5 = np.mean(abs_errors <= 5) * 100
        pearson = pearsonr(preds, labels)[0]
        spearman = spearmanr(preds, labels)[0]
        print(f"MAE: {mae:.2f} | std: {std_error:.2f}")
        print(f" CS@5: {cs5:.2f}%")
        print(f"[Test] Pearson: {pearson:.3f} | Spearman: {spearman:.3f}")

        true_ages = np.array(results["labels"]).reshape(-1)
        pred_ages = np.array(results["preds"]).reshape(-1)
        true_gender = np.array(results["gender_labels"]).reshape(-1)
        pred_gender = np.array(results["gender_preds"]).reshape(-1)

        test_output_dir = os.path.join(self.save_path, "test")
        os.makedirs(test_output_dir, exist_ok=True)

        plt.figure(figsize=(7, 7))
        plt.scatter(
            true_ages,
            pred_ages,
            color="tab:blue",
            edgecolor="k",
            alpha=0.5,
            linewidth=0.5,
            label="Prediction",
        )
        plt.plot(
            [true_ages.min(), true_ages.max()],
            [true_ages.min(), true_ages.max()],
            "gray",
            linestyle="--",
            label="y = x",
        )
        plt.xlabel("True Age")
        plt.ylabel("Predicted Age")
        plt.title("True vs Predicted Age")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        scatter_path = os.path.join(test_output_dir, "age_scatter.png")
        plt.savefig(scatter_path, dpi=300)
        plt.close()

        eval_df = pd.DataFrame(
            {
                "true_age": true_ages,
                "pred_age": pred_ages,
                "true_gender": true_gender,
                "pred_gender": pred_gender,
            }
        )
        eval_df["true_age"] = eval_df["true_age"].round(2)
        grouped = (
            eval_df.groupby("true_age")["pred_age"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped["std"] = grouped["std"].fillna(0)
        group_true = grouped["true_age"]
        group_pred_mean = grouped["mean"]

        group_errors = group_pred_mean - group_true
        group_abs_errors = np.abs(group_errors)

        g_mae = np.mean(group_abs_errors)
        g_std_error = np.std(group_errors)
        g_cs5 = np.mean(group_abs_errors <= 5) * 100
        g_pearson = pearsonr(group_pred_mean, group_true)[0]
        print("Grouped Evaluation")
        print(f"[Group] MAE: {g_mae:.2f}")
        print(f"[Group] std: {g_std_error:.2f}")
        print(f"[Group] CS@5: {g_cs5:.2f}%")
        print(f"[Group] Pearson: {g_pearson:.3f}")

        all_min = min(true_ages.min(), pred_ages.min())
        all_max = max(true_ages.max(), pred_ages.max())
        display_range = [all_min - 2, all_max + 2]
        plt.figure(figsize=(12, 8))
        unique_ages = np.sort(eval_df["true_age"].unique())
        male_df = eval_df[eval_df["true_gender"] == 0]
        female_df = eval_df[eval_df["true_gender"] == 1]
        male_plot_data = [
            male_df[male_df["true_age"] == age]["pred_age"].values
            for age in unique_ages
        ]
        female_plot_data = [
            female_df[female_df["true_age"] == age]["pred_age"].values
            for age in unique_ages
        ]
        width = 0.3
        offset = 0.2

        m_valid_indices = [i for i, data in enumerate(male_plot_data) if len(data) > 0]
        if m_valid_indices:
            bp_m = plt.boxplot(
                [male_plot_data[i] for i in m_valid_indices],
                positions=unique_ages[m_valid_indices] - offset,
                widths=width,
                manage_ticks=False,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp_m["boxes"]:
                patch.set(facecolor="lightblue", color="blue", alpha=0.5)
            for median in bp_m["medians"]:
                median.set(color="darkblue", linewidth=2)
            plt.plot([], [], color="blue", label="Male")

        f_valid_indices = [
            i for i, data in enumerate(female_plot_data) if len(data) > 0
        ]
        if f_valid_indices:
            bp_f = plt.boxplot(
                [female_plot_data[i] for i in f_valid_indices],
                positions=unique_ages[f_valid_indices] + offset,
                widths=width,
                manage_ticks=False,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp_f["boxes"]:
                patch.set(facecolor="pink", color="red", alpha=0.5)
            for median in bp_f["medians"]:
                median.set(color="darkred", linewidth=2)
            plt.plot([], [], color="red", label="Female")
        plt.scatter(
            male_df["true_age"] - offset,
            male_df["pred_age"],
            color="blue",
            alpha=0.15,
            s=10,
        )
        plt.scatter(
            female_df["true_age"] + offset,
            female_df["pred_age"],
            color="red",
            alpha=0.15,
            s=10,
        )

        plt.plot(
            display_range,
            display_range,
            "gray",
            linestyle="--",
            alpha=0.5,
            label="Ideal (y=x)",
        )
        plt.xlim(display_range)
        plt.ylim(display_range)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Chronological Age")
        plt.ylabel("Predicted Age")
        plt.title("Age Prediction Distribution by Gender")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"
        boxplot_path = os.path.join(test_output_dir, "age_gender_boxplot.png")
        plt.savefig(boxplot_path, dpi=300)
        plt.close()

        g_preds = results["gender_preds"]
        g_labels = results["gender_labels"]
        report = classification_report(
            g_labels, g_preds, target_names=["Male", "Female"], digits=4
        )
        print("Gender Classification Report")
        print(report)
        print("End Evaluation...")

    def trainer(self, phase, dataloader, epoch):
        results = dict(iter=0, loss=0)
        self.phase = phase
        if phase == "train":
            self.net.train()
            print("Start Iteration...")
        else:
            self.net.eval()
        for data in tqdm(dataloader):
            results["iter"] += 1
            images = data["image"]
            labels = data["age"]
            gender = data["gender"]

            images = images.to(self.device)
            labels = labels.float().to(self.device).unsqueeze(-1)
            gender = gender.to(self.device)

            pred, gender_pred = self.net(images)

            if isinstance(pred, tuple):
                pred = pred[0]

            loss_age = F.mse_loss(pred, labels, reduction="mean")
            loss_gender = self.cross(gender_pred, gender)
            loss = loss_age + (self.gender_loss_weight * loss_gender)

            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            results["loss"] += loss.item()
            if phase != "train":
                p = pred.detach().cpu().numpy().flatten()
                l = labels.detach().cpu().numpy().flatten()
                p_gen = torch.argmax(gender_pred, dim=1).detach().cpu().numpy().flatten()
                l_gen = gender.detach().cpu().numpy().flatten()

                if "preds" not in results:
                    results["preds"] = p
                    results["labels"] = l
                    results["gender_preds"] = p_gen
                    results["gender_labels"] = l_gen
                else:
                    results["preds"] = np.concatenate([results["preds"], p])
                    results["labels"] = np.concatenate([results["labels"], l])
                    results["gender_preds"] = np.concatenate(
                        [results["gender_preds"], p_gen]
                    )
                    results["gender_labels"] = np.concatenate(
                        [results["gender_labels"], l_gen]
                    )
        return results
