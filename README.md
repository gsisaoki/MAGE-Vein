# MAGE-Vein: Multi-Instance Age and Gender Estimation from Finger Vein Images

Official PyTorch implementation of **MAGE-Vein**, accepted at **IJCB 2026**.

MAGE-Vein performs joint **age regression** and **gender classification** from grouped finger-vein images. The model uses a DenseNet-161 backbone with a hybrid fusion strategy: per-finger features are combined with a group-average feature before multi-task prediction heads.

> **Note:** The finger-vein dataset used in the paper is **not publicly available**. You must prepare your own data following the CSV schema below.

## Features

- Multi-instance input: three finger images per sample (fixed to 3 images in the proposed model)
- Multi-task learning: age regression + binary gender classification
- Config-driven training and evaluation (`configs/default.yaml`)
- Best-checkpoint saving based on validation MAE

## Installation

Requires **Python 3.9+** and a CUDA-capable GPU (recommended). CPU execution is supported with automatic fallback when CUDA is unavailable.

```bash
# Clone the repository
git clone https://github.com/gsisaoki/MAGE-Vein.git
cd MAGE-Vein

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

```

## Pre-trained Models

You can download the pre-trained weights (`best.pth.tar`) from the official [GitHub Releases](https://github.com/gsisaoki/MAGE-Vein/releases/tag/v1.0.0).

Please create a `checkpoints/` directory under the project root and place the downloaded file there before running the evaluation script:

```bash
mkdir -p checkpoints
# Move the downloaded best.pth.tar into checkpoints/

```

## Data Preparation

Provide train, validation, and test CSV manifests and update the paths in `configs/default.yaml` under the `data:` section.

### CSV Schema

Each manifest must contain the following columns:

| Column | Type | Description |
| --- | --- | --- |
| `file_path` | str | Image path **relative to** `data.data_root` |
| `age` | int | Subject age in years (precomputed by the user) |
| `gender` | int | `0` = male, `1` = female |
| `group_id` | str | Shared ID for images from the same finger/session |

Rows with the same `group_id` are grouped together. Every consecutive 3 rows within a group form one training/evaluation sample.

**Example** (see also `data/example_train_list.csv`):

```csv
file_path,age,gender,group_id
path/to/dataset/subject001/img_001_0.bmp,34,0,subject001_L
path/to/dataset/subject001/img_001_1.bmp,34,0,subject001_L
path/to/dataset/subject001/img_001_2.bmp,34,0,subject001_L
path/to/dataset/subject002/img_002_0.bmp,52,1,subject002_R
path/to/dataset/subject002/img_002_1.bmp,52,1,subject002_R

```

Update `configs/default.yaml`:

```yaml
data:
  train_csv: path/to/dataset/train_list.csv
  val_csv:   path/to/dataset/val_list.csv
  test_csv:  path/to/dataset/test_list.csv
  data_root: path/to/dataset
  crop_size: 224
  images_per_group: 3

```

## Usage

### Training

```bash
python train.py \
  --config configs/default.yaml \
  --exp-name my_experiment \
  --device 0 \
  --num-workers 4

```

| Argument | Description |
| --- | --- |
| `--config` | Path to the YAML configuration file (default: `configs/default.yaml`) |
| `--exp-name` | **Required.** Experiment suffix; a date prefix is added automatically (e.g. `2026-07-03_my_experiment`) |
| `--device` | CUDA device index (`0`, `1`, …) or `cpu` (default: `0`) |
| `--num-workers` | Number of DataLoader workers (overrides config when set) |
| `--epochs` | Override training epochs from config |
| `--batch-size` | Override batch size from config |
| `--lr` | Override learning rate from config |

Outputs are written under `results/<exp_name>/`:

```
results/<exp_name>/
├── checkpoints/best.pth.tar
├── tensorboard/
├── results/
└── config.yaml

```

### Evaluation

Run evaluation with the **best** checkpoint saved during training:

```bash
python test.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth.tar \
  --device 0 \
  --num-workers 4

```

| Argument | Description |
| --- | --- |
| `--config` | Path to the YAML configuration file (default: `configs/default.yaml`) |
| `--checkpoint` | **Required.** Path to `best.pth.tar` from training or downloaded pre-trained models |
| `--device` | CUDA device index or `cpu` (default: `0`) |
| `--num-workers` | Number of DataLoader workers (overrides config when set) |
| `--exp-name` | Experiment name for output paths; inferred from `--checkpoint` when omitted |

Evaluation metrics (printed to stdout and figures saved under `results/<exp_name>/results/test/`):

* Age: MAE, CS@5, Pearson / Spearman correlation
* Gender: classification report (accuracy, precision, recall, F1)

## Project Structure

```
configs/default.yaml   # Hyperparameters and paths
data/example_*.csv     # Synthetic CSV format examples
src/
  dataset.py           # GroupedFingerVeinDataset
  models.py            # MAGEVein model and build_model()
  solver.py            # Training, validation, and test logic
  utils.py             # Config loading, checkpoints, utilities
train.py               # Training entry point
test.py                # Evaluation entry point

```

## Results

Age estimation accuracy on the private finger-vein dataset (IJCB 2026). Lower is better for MAE and Std.; higher is better for Corr. and CS@5.

| Model | MAE [y/o] ↓ | Corr. ↑ | CS@5 ↑ | Std. [y/o] ↓ |
| --- | --- | --- | --- | --- |
| Dataset Avg. | 13.61 | — | 0.184 | 16.08 |
| Wimmer et al. (ICPRW 2023) | 9.33 | 0.700 | 0.331 | 11.59 |
| **MAGE-Vein (Proposed)** | **6.47** | **0.876** | **0.455** | **8.05** |
| **MAGE-Vein w/ Aug. (Proposed w/ Aug.)** | **6.12** | **0.880** | **0.526** | **7.77** |

* **MAE:** Mean Absolute Error (years)
* **Corr.:** Correlation coefficient between predicted and chronological age
* **CS@5:** Cumulative Score — fraction of samples with absolute error ≤ 5 years
* **Std:** Standard deviation of prediction errors (years)

The best results are achieved with vertical-flip augmentation during training (`MAGE-Vein w/ Aug.`). This repository implements the proposed multi-instance architecture; enable training-time augmentation via `GroupedFingerVeinDataset` (random vertical flip, `p=0.4`).

## License

This project is released under a **research-use-only** license by **Computer Structures Laboratory, Graduate School of Information Sciences, Tohoku University**. See [LICENSE](https://github.com/gsisaoki/MAGE-Vein/blob/main/LICENSE) for details. Commercial use requires prior written permission from the copyright holder.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{MAGE-Vein2026,
  author    = "Tanaka, K. and Ito, K. and Aoki, T. and Fujio, M. and Kaga, Y. and Oshima, K. and Takahashi, K.",
  title     = "{MAGE-Vein}: {M}ulti-Instance Age and Gender Estimation from Finger Vein Images",
  booktitle = "Proc. IEEE Int. J. Conf. Biometrics",
  year      = "2026",
  month     = sep
}

```

## Contact

For questions about the paper or code, please open a GitHub issue or contact the authors listed in the publication.
