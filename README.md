# EmoCaps-PT

EEG-based emotion recognition and vigilance detection using **Efficient Capsule Networks (EfficientCapsNet)**, implemented in PyTorch.

Supports two datasets:
- **DEAP** — emotion recognition across Valence / Arousal / Dominance / Liking / VA (4-class) / VAD (8-class)
- **SEED-VIG** — vigilance / drowsiness detection (Awake / Tired / Drowsy) via PERCLOS

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### DEAP

1. Apply for the [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and download `data_preprocessed_python.zip`
2. Extract to the path set in `configs/DEAP.yaml` → `data_path`
3. Set `preprocessed_path` in `configs/DEAP.yaml` to an empty directory where `.hdf` cache files will be written on first load

### SEED-VIG

1. Obtain the [SEED-VIG dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html)
2. Set `data_path` in `configs/SEEDVIG.yaml` to the directory containing `DE/` and `perclos_labels/`

---

## Training

The priorities of configuration values are: **command line > config file > argparse defaults**.

**SEED-VIG** (subject-independent, 10-fold CV):
```bash
python main.py --config configs/baseline.yaml \
               --dataset seedvig \
               --device 0 \
               --phase train
```

**DEAP subject-independent** (all subjects pooled, 10-fold CV):
```bash
python main.py --config configs/baseline.yaml \
               --dataset deap \
               --mode sub_independent \
               --device 0 \
               --phase train
```

**DEAP subject-dependent** (one model per subject, 10-fold CV):
```bash
python main.py --dataset deap \
               --mode sub_dependent \
               --device 0 \
               --phase train
```

> When `--work-dir` is set, all outputs (models, logs, results) are redirected under that directory and a `config.yaml` snapshot is saved there for reproducibility.

---

## Inference

We provide pretrained models for inference. Download links:

| Dataset | Backbone | Accuracy | Pretrained model |
|---------|----------|----------|-----------------|
| SEED-VIG | EfficientCapsNet | — | *(coming soon)* |
| DEAP (VA) | EfficientCapsNet | — | *(coming soon)* |

To evaluate a pretrained model:

```bash
# SEED-VIG
python main.py --load-weights PATH_TO_PRETRAINED_MODEL \
               --dataset seedvig \
               --phase test

# DEAP — VA dimension, subject-independent
python main.py --load-weights PATH_TO_PRETRAINED_MODEL \
               --dataset deap \
               --dimension VA \
               --phase test

# DEAP — subject-dependent, subject 1, VAD dimension
python main.py --load-weights PATH_TO_PRETRAINED_MODEL \
               --dataset deap \
               --mode sub_dependent \
               --subject 1 \
               --dimension VAD \
               --phase test
```

The test phase prints per-class **Precision / Recall / F1 / Support** and overall accuracy.

---

## Feature Extraction

Capsule-vector features can be extracted from a trained model for downstream analysis (e.g. t-SNE, transfer learning):

```bash
python main.py --load-weights PATH_TO_PRETRAINED_MODEL \
               --dataset seedvig \
               --work-dir runs/exp1 \
               --phase features
```

Features are saved to `{work-dir}/features/` (or `./features/` when `--work-dir` is omitted):

```python
import numpy as np
data = np.load('features/SEEDVIG_features.npy', allow_pickle=True).item()
data['features']  # ndarray [N, num_class, 8]  — capsule vectors (FCCaps output)
data['lengths']   # ndarray [N, num_class]      — class probabilities
data['labels']    # ndarray [N]                 — integer ground-truth labels
```

---

## Project Structure

```
emocaps-pt/
│
├── main.py                        # Unified entry point (train / test / features)
├── eeg_scripts.py                 # Experiment functions: eeg_train, eeg_eval, eeg_features
├── eeg_network.py                 # Network architectures: capsule layers + DEAP/SEEDVIG models
│
├── configs/
│   ├── baseline.yaml              # Shared hyperparameters (epochs, lr, folds, output dirs …)
│   ├── DEAP.yaml                  # DEAP-specific settings (paths, input shape, dimensions)
│   └── SEEDVIG.yaml               # SEED-VIG-specific settings (paths, channels, thresholds)
│
└── utils/
    ├── __init__.py
    ├── pre_process_deap.py        # DEAP data loading, normalisation, DataLoader construction
    ├── pre_process_seedvig.py     # SEED-VIG DE feature loading, PERCLOS discretisation
    └── tools.py                   # margin_loss, get_save_path, save_best_model
```

### File descriptions

| File | Description |
|------|-------------|
| `main.py` | Argument parsing, config loading, device setup, dispatch to `eeg_scripts`. |
| `eeg_scripts.py` | `eeg_train` — K-fold CV loop (DEAP / SEEDVIG). `eeg_eval` — inference + per-class metrics. `eeg_features` — capsule-vector extraction. `CapsNetTrainer` — Adam + LR decay + early stopping training wrapper. |
| `eeg_network.py` | `PrimaryCaps`, `FCCaps`, `Length`, `Mask` capsule layers. `EfficientCapsNetDEAP` (Conv×3 → PrimaryCaps → FCCaps → Length). `EfficientCapsNetSEEDVIG` (Conv×2 → PrimaryCaps → FCCaps → Length). |
| `configs/baseline.yaml` | Shared defaults: epochs, batch size, lr, lr decay, early-stop patience, num folds, output directory names. |
| `configs/DEAP.yaml` | DEAP paths, input shape `[128, 32, 1]`, dimension list, num classes per dimension. |
| `configs/SEEDVIG.yaml` | SEED-VIG paths, input shape `[17, 5, 1]`, channel selection, PERCLOS thresholds. |
| `utils/pre_process_deap.py` | `DataDEAP` class — loads per-subject `.hdf` files, windowing, z-score normalisation, 1D→2D electrode grid mapping. `generate_data_loaders` — wraps data into PyTorch `DataLoader`. |
| `utils/pre_process_seedvig.py` | `DataSEEDVIG` class — loads per-session `.mat` DE features and PERCLOS labels, channel selection, PERCLOS discretisation. `generate_seedvig_loaders`. |
| `utils/tools.py` | `margin_loss` (capsule margin loss from Sabour et al.), `get_save_path` (structured output path builder), `save_best_model` (keeps highest-accuracy fold checkpoint). |

---

## Output

| Path | Contents |
|------|----------|
| `{work-dir}/bin/` | Per-fold and best-model weight files (`.pt`) |
| `{work-dir}/csv_logs/` | Per-fold training curves — epoch, loss, accuracy (`.csv`) |
| `{work-dir}/results/` | `summary.csv` (fold-level accuracy) + `hyperparam.csv` (10-fold averages) |
| `{work-dir}/features/` | Extracted capsule vectors (`.npy`) from `--phase features` |
| `{work-dir}/config.yaml` | Snapshot of the merged config used for the run |

---

## Model Architecture

### EfficientCapsNetDEAP — input `[batch, 128, 32, 1]`

```
permute → [batch, 1, 128, 32]
  Conv2d(5×5, 32ch) → BN → ReLU
  Conv2d(3×3, 64ch) → BN → ReLU
  Conv2d(3×3, 128ch, stride=2) → BN → ReLU   → [batch, 128, 64, 16]
  PrimaryCaps (depthwise 9×9)                  → [batch, 7168, 8]
  FCCaps (self-attention routing)               → [batch, num_class, 8]
  Length                                        → [batch, num_class]
```

### EfficientCapsNetSEEDVIG — input `[batch, n_ch, 5, 1]`

```
permute → [batch, 1, n_ch, 5]
  Conv2d(3×3, 64ch) → BN → ReLU
  Conv2d(3×3, 64ch) → BN → ReLU
  PrimaryCaps (depthwise 3×3)                  → [batch, N, 8]
  FCCaps (self-attention routing)               → [batch, num_class, 8]
  Length                                        → [batch, num_class]
```

---

## Citation

If you use this code, please cite the original EfficientCapsNet paper:

```bibtex
@article{mazzia2021efficient,
  title={Efficient-CapsNet: Capsule Network with Self-Attention Routing},
  author={Mazzia, Vittorio and Salvetti, Francesco and Chiaberge, Marcello},
  journal={Scientific Reports},
  year={2021}
}
```
