import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def margin_loss(y_true, y_pred):
    lbd = 0.5
    m_plus = 0.9
    m_minus = 0.1

    L = y_true * F.relu(m_plus - y_pred) ** 2 + \
        lbd * (1 - y_true) * F.relu(y_pred - m_minus) ** 2
    return L.sum(dim=-1).mean()


def get_save_path(save_dir, dimension=None, subject=None):
    new_path = Path(save_dir)
    if subject is not None:
        new_path = new_path / f"sub_{subject}"
    if dimension:
        new_path = new_path / f"{dimension}"
    new_path.mkdir(parents=True, exist_ok=True)
    return new_path


def save_best_model(save_dir, test_results):
    best_fold_result = np.argmax(test_results)
    p = Path(save_dir / f"fold_{best_fold_result}.pt")
    if p.exists():
        p.replace(Path(p.parent, f"best_model{p.suffix}"))
    for child in save_dir.glob('fold_*.pt'):
        child.unlink(missing_ok=True)
