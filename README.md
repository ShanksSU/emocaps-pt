# EmoCaps-PT

EEG-based emotion recognition and vigilance detection using **Efficient-CapsNet** with self-attention routing, implemented in **PyTorch**.

> This repo is based on [Efficient-CapsNet](http://nature.com/articles/s41598-021-93977-0) and [EmoCaps-Tf](https://github.com/Aadimator/EmoCaps-Tf). Many thanks for their great work!

This project adapts the Efficient-CapsNet architecture (Mazzia et al., 2021) from image classification to EEG signal analysis, supporting two benchmark datasets:

| Dataset | Task | Labels |
|---------|------|--------|
| **DEAP** | Emotion recognition | Valence / Arousal / Dominance / Liking / VA (4-class) / VAD (8-class) |
| **SEED-VIG** | Vigilance detection | Awake / Tired / Drowsy (PERCLOS-based) |

---

## Requirements

The recommended way is to use the pre-built Docker image, which already includes PyTorch 2.2.2 and CUDA 12.8.1:

```bash
docker pull dsu52062/pytorch-devel:2.2.2-cuda12.8.1-ubuntu22.04
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    dsu52062/pytorch-devel:2.2.2-cuda12.8.1-ubuntu22.04
```

Then inside the container, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `numpy==1.26.4`, `pandas`, `scikit-learn`, `h5py`, `tqdm`, `pyyaml`, `matplotlib`, `openpyxl`.

---

## Dataset Setup

### DEAP

1. Apply for access at [https://www.kaggle.com/datasets/manh123df/deap-dataset](https://www.kaggle.com/datasets/manh123df/deap-dataset) and download `data_preprocessed_python.zip`
2. Extract all 32 subject `.dat` files into a single directory
3. Edit `configs/DEAP.yaml`:
   ```yaml
   data_path: "/path/to/data_preprocessed_python/"
   preprocessed_path: "/path/to/empty/cache/dir/"   # HDF5 cache, created on first run
   ```

The preprocessor applies a 128-sample sliding window, z-score normalisation, and maps the 32 electrodes onto a 2-D spatial grid `[128, 32, 1]`.

### SEED-VIG

1. Obtain the dataset at [https://www.kaggle.com/datasets/mojahidmahin/seed-vig](https://www.kaggle.com/datasets/mojahidmahin/seed-vig)
2. Ensure the directory contains `DE/` (differential entropy features) and `perclos_labels/`
3. Edit `configs/SEEDVIG.yaml`:
   ```yaml
   data_path: "/path/to/seed-vig/"
   ```

Each sample is a `[channels, 5, 1]` tensor of DE features across 5 frequency bands (δ, θ, α, β, γ). PERCLOS thresholds define class boundaries.

---

## Training

### SEED-VIG

```bash
python main.py --config configs/baseline.yaml \
               --dataset seedvig \
               --device 0 \
               --phase train \
               --work-dir runs/seedvig_exp1
```

### DEAP - subject-independent (all subjects pooled)

```bash
python main.py --config configs/baseline.yaml \
               --dataset deap \
               --mode sub_independent \
               --device 0 \
               --phase train \
               --work-dir runs/deap_sub_ind
```

### DEAP - subject-dependent (one model per subject)

```bash
python main.py --dataset deap \
               --mode sub_dependent \
               --device 0 \
               --phase train \
               --work-dir runs/deap_sub_dep
```

> Setting `--work-dir` redirects all outputs (checkpoints, logs, results) under that path and saves a `config.yaml` snapshot for full reproducibility.

---

## Inference

```bash
# SEED-VIG
python main.py --load-weights PATH/TO/best_model.pt \
               --dataset seedvig \
               --phase test

# DEAP - VA dimension, subject-independent
python main.py --load-weights PATH/TO/best_model.pt \
               --dataset deap \
               --dimension VA \
               --phase test

# DEAP - subject-dependent, subject 3, VAD dimension
python main.py --load-weights PATH/TO/best_model.pt \
               --dataset deap \
               --mode sub_dependent \
               --subject 3 \
               --dimension VAD \
               --phase test
```

The test phase prints per-class **Precision / Recall / F1 / Support** and overall accuracy, and saves results to `.txt` and `.csv` next to the weight file.

---

### Module descriptions

| File | Role |
|------|------|
| `main.py` | Parses arguments, merges YAML configs, dispatches to `eeg_scripts`. |
| `eeg_network.py` | Defines `PrimaryCaps`, `FCCaps`, `Length`, `Mask`, `Decoder` layers and the two EEG model classes. |
| `eeg_scripts.py` | `CapsNetTrainer` (Adam + LR decay + early stopping). `eeg_train` (K-fold CV). `eeg_eval` (per-class metrics). `eeg_features` (capsule vector export). |
| `configs/baseline.yaml` | Training defaults shared by all datasets. |
| `configs/DEAP.yaml` | DEAP input shape `[128, 32, 1]`, dimension/class mapping, file paths. |
| `configs/SEEDVIG.yaml` | SEED-VIG input shape, electrode channel selection, PERCLOS thresholds. |
| `utils/pre_process_deap.py` | `DataDEAP` - loads per-subject `.dat` files, sliding window, z-score, electrode grid. `generate_data_loaders` - returns `DataLoader` pairs. |
| `utils/pre_process_seedvig.py` | `DataSEEDVIG` - loads session-level `.mat` DE features and PERCLOS labels, channel selection. |
| `utils/tools.py` | `margin_loss`, `get_save_path`, `save_best_model`. |

---

## Output Files

| Path | Contents |
|------|----------|
| `{work-dir}/bin/fold_N.pt` | Per-fold model checkpoint |
| `{work-dir}/bin/best_model.pt` | Checkpoint of the highest-accuracy fold |
| `{work-dir}/csv_logs/fold_N.csv` | Training curve per fold (epoch, loss, accuracy) |
| `{work-dir}/results/summary.csv` | Fold-level test accuracy and timing |
| `{work-dir}/results/hyperparam.csv` | 10-fold mean accuracy and mean timing |
| `{work-dir}/features/*.npy` | Extracted capsule vectors (`--phase features`) |
| `{work-dir}/config.yaml` | Merged config snapshot for reproducibility |
| `{weight_dir}/*_eval.txt` | Per-class precision/recall/F1 report (`--phase test`) |
| `{weight_dir}/*_eval.csv` | Same report in tabular form |

---

## Citation

If you use this repository, please cite the original Efficient-CapsNet paper:

```bibtex
@article{mazzia2021efficient,
  title={Efficient-CapsNet: capsule network with self-attention routing},
  author={Mazzia, Vittorio and Salvetti, Francesco and Chiaberge, Marcello},
  year={2021},
  journal={Scientific reports},
  publisher={Nature Publishing Group},
  volume={11}
}
```
