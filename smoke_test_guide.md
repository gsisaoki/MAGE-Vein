# MAGE-Vein スモークテスト手順書

このドキュメントは、MAGE-Vein リポジトリのコードが正しく動作するかを **最小構成（1 エポック）** で確認するための手順書です。  
本番の学習を行う前に、環境構築とパイプライン（学習 → 評価）がエラーなく完走するかを確認してください。

---

## 前提条件

- **Python 3.9 以上** を推奨します。
- リポジトリのルートディレクトリで作業してください。
- `data/example_*.csv` は **CSV 形式のサンプル** です。**画像ファイルは含まれていません。**
  - スモークテストを行う前に、CSV が参照するパスに合わせて **ダミー BMP 画像を生成する必要があります**（Step B）。
- ダミーデータの 3 枚組有効サンプルは **3 件** と少ないため、学習時は **`--batch-size 1` を必ず指定** してください。

---

## Step A: 環境構築

仮想環境を作成し、依存パッケージをインストールします。

```bash
cd /path/to/Age_Estimation_Fingervein

python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

> 既に `.venv` がある場合は、`source .venv/bin/activate` のみ実行してから Step B 以降に進んでください。

---

## Step B: テスト用ダミー画像の生成

以下の Python スクリプトを **そのままコピー＆ペースト** して実行してください。  
`smoke_data/` 以下に、CSV が参照する 9 枚のダミー BMP 画像を生成します。

```bash
python3 << 'PYEOF'
import os
import cv2
import numpy as np

files = [
    "path/to/dataset/subject001/img_001_0.bmp",
    "path/to/dataset/subject001/img_001_1.bmp",
    "path/to/dataset/subject001/img_001_2.bmp",
    "path/to/dataset/subject002/img_002_0.bmp",
    "path/to/dataset/subject002/img_002_1.bmp",
    "path/to/dataset/subject002/img_002_2.bmp",
    "path/to/dataset/subject003/img_003_0.bmp",
    "path/to/dataset/subject003/img_003_1.bmp",
    "path/to/dataset/subject003/img_003_2.bmp",
]

data_root = "smoke_data"
for rel_path in files:
    out_path = os.path.join(data_root, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    cv2.imwrite(out_path, img)
    print("Created:", out_path)
PYEOF
```

**補足**

- `example_train_list.csv` では 3 枚組グループが **3 サンプル** 成立します（男性 1 グループ + 女性 2 グループ）。
- 学習時は **`--batch-size 1` を必ず指定** してください（デフォルトの `batch_size: 16` では DataLoader が空になり、学習が正常に進みません）。

---

## Step C: テスト専用設定ファイルの作成

以下のコマンドで、スモークテスト用の最小構成 YAML を生成します。

```bash
cat > configs/smoke_test.yaml << 'EOF'
seed: 20
output_dir: results/
checkpoint_path: null

device: cpu
num_workers: 0
exp_name: null

data:
  train_csv: data/example_train_list.csv
  val_csv: data/example_val_list.csv
  test_csv: data/example_test_list.csv
  data_root: smoke_data
  crop_size: 224
  images_per_group: 3

model:
  name: multi
  pretrained: false

train:
  epochs: 1
  optimizer: adamw
  lr: 0.0001
  weight_decay: 0.01
  batch_size: 1
  gender_loss_weight: 20.0
  early_stop_patience: 10
EOF
```

| 設定項目 | 値 | 理由 |
|----------|-----|------|
| `pretrained: false` | ImageNet 重みを読み込まない | 初回テストを高速化 |
| `num_workers: 0` | ワーカー数 0 | CPU / 小規模データ向け、マルチプロセス問題を回避 |
| `batch_size: 1` | バッチサイズ 1 | ダミーデータ（3 サンプル）の小規模構成に対応 |
| `epochs: 1` | 1 エポック | 動作確認のみ |

---

## Step D: 学習テスト（1 エポック）

### CPU 環境の場合

```bash
python train.py \
  --config configs/smoke_test.yaml \
  --exp-name smoke_test \
  --device cpu \
  --num-workers 0 \
  --epochs 1 \
  --batch-size 1
```

### GPU 環境の場合

```bash
python train.py \
  --config configs/smoke_test.yaml \
  --exp-name smoke_test \
  --device 0 \
  --num-workers 2 \
  --epochs 1 \
  --batch-size 1
```

| 引数 | 説明 |
|------|------|
| `--config` | 使用する設定ファイル（Step C で作成した YAML） |
| `--exp-name` | 実験名（日付プレフィックスが自動付与されます） |
| `--device` | `cpu` または GPU 番号（`0`, `1`, …） |
| `--num-workers` | DataLoader のワーカー数 |
| `--epochs` | エポック数（1 で十分） |
| `--batch-size` | バッチサイズ（**必ず 1**） |

### 成功の目安

- コンソールに `Start Training...` → `End Training` が表示される
- 以下のチェックポイントが生成される:

```
results/<日付>_smoke_test/checkpoints/best.pth.tar
```

例: `results/2026-07-03_smoke_test/checkpoints/best.pth.tar`

---

## Step E: 評価テスト

Step D で生成された **ベストチェックポイント** を指定して評価を実行します。  
`<日付>` は Step D の出力に合わせて置き換えてください。

### CPU 環境の場合

```bash
python test.py \
  --config configs/smoke_test.yaml \
  --checkpoint results/2026-07-03_smoke_test/checkpoints/best.pth.tar \
  --device cpu \
  --num-workers 0
```

### GPU 環境の場合

```bash
python test.py \
  --config configs/smoke_test.yaml \
  --checkpoint results/2026-07-03_smoke_test/checkpoints/best.pth.tar \
  --device 0 \
  --num-workers 2
```

| 引数 | 説明 |
|------|------|
| `--config` | Step C と同じ設定ファイル |
| `--checkpoint` | Step D で保存された `best.pth.tar` のパス（**必須**） |
| `--device` | `cpu` または GPU 番号 |
| `--num-workers` | DataLoader のワーカー数 |

### 成功の目安

- コンソールに MAE、CS@5、Pearson / Spearman 相関、Gender Classification Report が表示される
- `End Evaluation...` まで完走する
- 以下の図が保存される:

```
results/<日付>_smoke_test/results/test/age_scatter.png
results/<日付>_smoke_test/results/test/age_gender_boxplot.png
```

---

## Step F: 後片付け（任意）

スモークテスト用のデータと出力を削除する場合:

```bash
rm -rf smoke_data results/*_smoke_test configs/smoke_test.yaml
```

---

## よくあるエラーと対処法

| エラー・症状 | 原因 | 対処法 |
|--------------|------|--------|
| `FileNotFoundError: Cannot read image: ...` | Step B のダミー画像生成が未実施、または `data_root` と画像パスの不一致 | Step B を再実行し、`configs/smoke_test.yaml` の `data.data_root: smoke_data` を確認する |
| DataLoader が空 / 学習ループが回らない | `batch_size` がサンプル数（3）より大きい | `--batch-size 1` を指定する（Step C の YAML も確認） |
| CUDA 関連エラー（GPU 非搭載環境） | GPU 指定で CUDA が利用できない | `--device cpu` を指定する |
| 初回起動が非常に遅い | `pretrained: true` で ImageNet 重みをダウンロードしている | `configs/smoke_test.yaml` で `pretrained: false` に設定する（Step C 参照） |

---

## 参考

- 本番学習・評価の詳細: [README.md](README.md)
- CSV スキーマ: `data/example_train_list.csv`
