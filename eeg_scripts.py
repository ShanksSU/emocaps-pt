import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from eeg_network import EfficientCapsNetDEAP, EfficientCapsNetSEEDVIG
from utils.pre_process_deap    import DataDEAP, pre_process, generate_data_loaders
from utils.pre_process_seedvig import DataSEEDVIG, pre_process_seedvig, generate_seedvig_loaders
from utils.tools               import get_save_path, margin_loss, save_best_model


# DEAP dimension metadata
class Dimensions(Enum):
    VALENCE   = "Valence"
    AROUSAL   = "Arousal"
    DOMINANCE = "Dominance"
    LIKING    = "Liking"
    VA        = "VA"
    VAD       = "VAD"

_DIM_NAME_MAP = {d.name: d for d in Dimensions}

_DEAP_DIM_NUM_CLASS = {
    'VALENCE': 2, 'AROUSAL': 2, 'DOMINANCE': 2, 'LIKING': 2, 'VA': 4, 'VAD': 8,
}
_DEAP_DIM_INDEX = {
    'VALENCE': 0, 'AROUSAL': 1, 'DOMINANCE': 2, 'LIKING': 3, 'VA': 4, 'VAD': 5,
}
_DEAP_CLASS_NAMES = {
    'VALENCE':   ['Low-V',  'High-V'],
    'AROUSAL':   ['Low-A',  'High-A'],
    'DOMINANCE': ['Low-D',  'High-D'],
    'LIKING':    ['Low-L',  'High-L'],
    'VA':        ['HVHA', 'HVLA', 'LVHA', 'LVLA'],
    'VAD':       ['HHH', 'HHL', 'HLH', 'HLL', 'LHH', 'LHL', 'LLH', 'LLL'],
}
_SEEDVIG_CLASS_NAMES = {1: ['Alert', 'Drowsy'], 2: ['Awake', 'Tired', 'Drowsy']}


# Internal helpers
def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _load_deap_data(args, config):
    """Return (data_np, labels_int) for the requested DEAP subset."""
    dim       = (args.dimension or 'VA').upper()
    num_class = _DEAP_DIM_NUM_CLASS[dim]
    dim_idx   = _DEAP_DIM_INDEX[dim]
    config['num_class'] = num_class

    deap = DataDEAP(config)
    if args.mode == 'sub_dependent':
        sub = (args.subject or 1) - 1
        print(f"Subject   : {sub + 1}")
        data_rnn, _, labels = deap.load_per_subject(sub)
    else:
        print("Mode      : subject-independent (all subjects pooled)")
        all_data, all_labels = [], []
        for s in range(config['subjects']):
            d, _, l = deap.load_per_subject(s)
            all_data.append(d)
            all_labels.append(l)
        data_rnn = np.concatenate(all_data,   axis=0)
        labels   = np.concatenate(all_labels, axis=0)

    selected_label = labels[:, dim_idx]
    data_np, _     = pre_process(data_rnn, selected_label, num_class)
    labels_int     = selected_label.astype(int)
    return dim, num_class, data_np, labels_int


def _load_seedvig_data(args, config):
    """Return (data_np, labels) for the requested SEED-VIG subset."""
    thresholds          = config.get('perclos_thresholds', [0.35, 0.70])
    num_class           = len(thresholds) + 1
    config['num_class'] = num_class

    ds = DataSEEDVIG(config)
    if args.session:
        eeg, labels = ds.load_session(args.session)
    else:
        print("Mode      : all sessions pooled")
        all_eeg, all_labels = [], []
        for sess in ds.sessions:
            e, l = ds.load_session(sess)
            all_eeg.append(e)
            all_labels.append(l)
        eeg    = np.concatenate(all_eeg,    axis=0)
        labels = np.concatenate(all_labels, axis=0)

    data_np, _ = pre_process_seedvig(eeg, labels, num_class)
    return num_class, data_np, labels


def _build_model(model_name, config, device, weight_path):
    if model_name == 'DEAP':
        model = EfficientCapsNetDEAP(
            config['input_shape'],
            config['num_class'],
            num_channels=config.get('num_channels', 32),
        )
    else:
        model = EfficientCapsNetSEEDVIG(config['input_shape'], config['num_class'])
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()
    print(f"Loaded    : {weight_path}")
    return model


def _batch_infer(model, data_np, device, batch_size):
    tensor = torch.from_numpy(data_np)
    preds  = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            _, caps_len = model(tensor[i : i + batch_size].to(device))
            preds.append(caps_len.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def _batch_extract(model, data_np, device, batch_size):
    """Return (capsule_vectors [N, num_class, D], capsule_lengths [N, num_class])."""
    tensor   = torch.from_numpy(data_np)
    all_vecs = []
    all_lens = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            caps_vec, caps_len = model(tensor[i : i + batch_size].to(device))
            all_vecs.append(caps_vec.cpu().numpy())
            all_lens.append(caps_len.cpu().numpy())
    return np.concatenate(all_vecs), np.concatenate(all_lens)


def _print_report(preds, labels_int, class_names):
    """Print metrics and return (report_lines, per_class_rows) for saving."""
    correct = (preds == labels_int).sum()
    total   = len(preds)
    header  = f"\nAccuracy: {correct}/{total} = {correct / total:.4f}\n"
    col_hdr = f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    sep     = '-' * 56

    lines = [header, col_hdr, sep]
    rows  = []
    for c, name in enumerate(class_names):
        tp   = ((preds == c) & (labels_int == c)).sum()
        fp   = ((preds == c) & (labels_int != c)).sum()
        fn   = ((preds != c) & (labels_int == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int((labels_int == c).sum())
        line = f"{name:<12} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}"
        lines.append(line)
        rows.append({'class': name, 'precision': prec, 'recall': rec,
                     'f1': f1, 'support': support})

    for line in lines:
        print(line)
    return lines, rows, correct / total


# CapsNetTrainer - training / evaluation wrapper
class CapsNetTrainer:
    def __init__(self, model_name, config, subject=None, fold=None, dimension=None,
                 mode='test', custom_path=None, verbose=False, desc=''):
        self.model_name = model_name
        self.config     = config
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        base_path = get_save_path(
            config['saved_model_dir'],
            dimension=dimension, subject=subject,
        )
        self.model_path      = custom_path if custom_path else base_path / 'best_model.pt'
        self.model_path_fold = base_path / f'fold_{fold}.pt'
        self.csv_log_path    = get_save_path(
            config['csv_log_save_dir'],
            dimension=dimension, subject=subject,
        ) / f'fold_{fold}.csv'

        self.model = self._build_model()
        if verbose:
            print(self.model)

    def _build_model(self):
        if self.model_name == 'DEAP':
            m = EfficientCapsNetDEAP(
                self.config['input_shape'],
                self.config['num_class'],
                num_channels=self.config.get('num_channels', 32),
            )
        elif self.model_name == 'SEEDVIG':
            m = EfficientCapsNetSEEDVIG(
                self.config['input_shape'],
                self.config['num_class'],
            )
        else:
            raise ValueError(f"Unknown model_name: {self.model_name!r}")
        return m.to(self.device)

    def train(self, train_loader, initial_epoch=0):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        base_lr   = self.config['lr']
        lr_dec    = self.config['lr_dec']
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(lr_dec ** epoch, 5e-5 / base_lr),
        )

        best_loss  = float('inf')
        no_improve = 0
        patience   = self.config.get('early_stop_patience', 4)
        records    = []

        print('-' * 30 + f' {self.model_name} train ' + '-' * 30)
        epoch_bar = tqdm(range(initial_epoch, self.config['epochs']), desc='Epochs', file=sys.stderr)

        for epoch in epoch_bar:
            self.model.train()
            total_loss = total_acc = n_batches = 0

            # batch_bar = tqdm(train_loader, desc=f'  Epoch {epoch + 1}', leave=False, file=sys.stderr)
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                _, caps_len = self.model(X)
                loss = margin_loss(y, caps_len)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_acc  += (caps_len.argmax(-1) == y.argmax(-1)).float().mean().item()
                n_batches  += 1
                # batch_bar.set_postfix(loss=f'{loss.item():.4f}')

            epoch_loss = total_loss / n_batches
            epoch_acc  = total_acc  / n_batches
            scheduler.step()
            epoch_bar.set_postfix(loss=f'{epoch_loss:.4f}', acc=f'{epoch_acc:.4f}')
            # print(f'Epoch {epoch + 1}/{self.config["epochs"]} '
            #       f'- loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}')
            records.append({'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': epoch_acc})

            if epoch_loss < best_loss - 1e-4:
                best_loss  = epoch_loss
                torch.save(self.model.state_dict(), self.model_path_fold)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(self.csv_log_path, index=False)
        return records

    def evaluate(self, test_loader):
        print('-' * 30 + f' {self.model_name} Evaluation ' + '-' * 30)
        self.model.eval()
        total_loss = total_acc = n_batches = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                _, caps_len = self.model(X)
                total_loss += margin_loss(y, caps_len).item()
                total_acc  += (caps_len.argmax(-1) == y.argmax(-1)).float().mean().item()
                n_batches  += 1

        avg_loss = total_loss / n_batches
        avg_acc  = total_acc  / n_batches
        print(f'Test loss: {avg_loss:.4f} - Test acc: {avg_acc:.4f}')
        return avg_loss, avg_acc


# eeg_train - K-fold cross-validation training
def eeg_train(config, mode='sub_independent'):
    dataset = config['dataset'].upper()
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if dataset == 'DEAP':
        _train_deap(config, mode)
    elif dataset == 'SEEDVIG':
        _train_seedvig(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")


def _train_deap(config, mode):
    model_name          = 'DEAP'
    dimension_index     = {_DIM_NAME_MAP[k]: v for k, v in config['dimension_index'].items()}
    dimension_num_class = {_DIM_NAME_MAP[k]: v for k, v in config['dimension_num_class'].items()}
    dimension_list      = [_DIM_NAME_MAP[n] for n in config['dimensions']]
    num_folds           = config['num_folds']
    kfold               = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                          random_state=config['random_seed'])
    deap                = DataDEAP(config)

    if mode == 'sub_dependent':
        for subject in range(config['subjects']):
            data_rnn, _, labels = deap.load_per_subject(subject)
            _deap_kfold(config, model_name, data_rnn, labels,
                        dimension_list, dimension_index, dimension_num_class,
                        kfold, num_folds, subject=subject, desc='')
    else:
        print("Loading all subjects...")
        all_data, all_labels = [], []
        for s in tqdm(range(config['subjects']), desc='Loading subjects', file=sys.stderr):
            d, _, l = deap.load_per_subject(s)
            all_data.append(d)
            all_labels.append(l)
        data_rnn = np.concatenate(all_data,   axis=0)
        labels   = np.concatenate(all_labels, axis=0)
        _deap_kfold(config, model_name, data_rnn, labels,
                    dimension_list, dimension_index, dimension_num_class,
                    kfold, num_folds, subject=None, desc='sub_independent')


def _deap_kfold(config, model_name, data_rnn, labels,
                dimension_list, dimension_index, dimension_num_class,
                kfold, num_folds, subject, desc):
    for selected_dim in dimension_list:
        print(f"\n{'=' * 10} {selected_dim.value} {'=' * 10}")
        selected_label      = labels[:, dimension_index[selected_dim]]
        config['num_class'] = dimension_num_class[selected_dim]

        test_acc_all   = np.zeros(num_folds)
        train_time_all = np.zeros(num_folds)
        test_time_all  = np.zeros(num_folds)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(data_rnn, selected_label)):
            tag = (f'Subject: {subject + 1}  Fold: {fold + 1}'
                   if subject is not None else f'Fold: {fold + 1}')
            print(f"\n{'=' * 10} {tag} {'=' * 10}")

            data, label = pre_process(data_rnn, selected_label, config['num_class'])
            train_loader, test_loader = generate_data_loaders(
                data[train_ids], label[train_ids],
                data[test_ids],  label[test_ids],
                config['batch_size'],
            )

            trainer = CapsNetTrainer(
                model_name, config,
                subject=subject, fold=fold, dimension=selected_dim,
                mode='train', desc=desc, verbose=False,
            )

            t0 = time.time()
            trainer.train(train_loader)
            train_time = time.time() - t0
            print(f'Train time: {_fmt_time(train_time)}')

            t1 = time.time()
            _, acc = trainer.evaluate(test_loader)
            test_time = time.time() - t1

            test_acc_all[fold]   = acc
            train_time_all[fold] = train_time
            test_time_all[fold]  = test_time
            torch.cuda.empty_cache()

        if subject is not None:
            save_best_model(
                get_save_path(config['saved_model_dir'],
                              dimension=selected_dim, subject=subject),
                test_acc_all,
            )

        result_path = get_save_path(config['results_dir'],
                                    dimension=selected_dim, subject=subject)
        pd.DataFrame({
            'fold':          range(1, num_folds + 1),
            'Test accuracy': test_acc_all,
            'train time':    train_time_all,
            'test time':     test_time_all,
        }).to_csv(result_path / 'summary.csv', index=False)
        pd.DataFrame({
            'average acc of 10 folds':        np.mean(test_acc_all),
            'average train time of 10 folds': np.mean(train_time_all),
            'average test time of 10 folds':  np.mean(test_time_all),
        }, index=['dimention/sub']).to_csv(result_path / 'hyperparam.csv', index=True)

        print(f'10-fold average accuracy:   {np.mean(test_acc_all):.4f}')
        print(f'10-fold average train time: {_fmt_time(np.mean(train_time_all))}')
        print(f'10-fold average test time:  {_fmt_time(np.mean(test_time_all))}')


def _train_seedvig(config):
    model_name          = 'SEEDVIG'
    num_class           = len(config['perclos_thresholds']) + 1
    config['num_class'] = num_class
    num_folds           = config['num_folds']
    kfold               = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                          random_state=config['random_seed'])
    desc                = 'sub_independent'
    class_names         = _SEEDVIG_CLASS_NAMES.get(
        len(config['perclos_thresholds']), [f'Class_{i}' for i in range(num_class)]
    )

    print(f"Classes ({num_class}): {class_names}")
    print(f"PERCLOS thresholds: {config['perclos_thresholds']}")

    ds = DataSEEDVIG(config)
    print(f"\nFound {len(ds.sessions)} sessions")
    all_eeg, all_labels = [], []
    for sess in tqdm(ds.sessions, desc='Sessions', file=sys.stderr):
        eeg, lbl = ds.load_session(sess)
        all_eeg.append(eeg)
        all_labels.append(lbl)

    eeg_all    = np.concatenate(all_eeg,    axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    print(f"Total samples: {len(eeg_all)}")
    print(f"Class distribution: {dict(zip(*np.unique(labels_all, return_counts=True)))}")

    test_acc_all   = np.zeros(num_folds)
    train_time_all = np.zeros(num_folds)
    test_time_all  = np.zeros(num_folds)

    print(f"\n{'=' * 10} SEED-VIG {num_folds}-fold CV {'=' * 10}")

    for fold, (train_ids, test_ids) in enumerate(kfold.split(eeg_all, labels_all)):
        print(f"\n{'=' * 10} Fold {fold + 1}/{num_folds} {'=' * 10}")
        print(f"  Train: {len(train_ids)}  Test: {len(test_ids)}")

        X, y = pre_process_seedvig(eeg_all, labels_all, num_class)
        train_loader, test_loader = generate_seedvig_loaders(
            X[train_ids], y[train_ids],
            X[test_ids],  y[test_ids],
            config['batch_size'],
        )

        trainer = CapsNetTrainer(
            model_name, config,
            fold=fold, mode='train', desc=desc, verbose=False,
        )

        t0 = time.time()
        trainer.train(train_loader)
        train_time = time.time() - t0
        print(f'Train time: {_fmt_time(train_time)}')

        t1 = time.time()
        _, acc = trainer.evaluate(test_loader)
        test_time = time.time() - t1

        test_acc_all[fold]   = acc
        train_time_all[fold] = train_time
        test_time_all[fold]  = test_time
        torch.cuda.empty_cache()

    result_path = get_save_path(config['results_dir'])
    pd.DataFrame({
        'fold':          range(1, num_folds + 1),
        'Test accuracy': test_acc_all,
        'train time':    train_time_all,
        'test time':     test_time_all,
    }).to_csv(result_path / 'summary.csv', index=False)
    pd.DataFrame({
        'average acc of 10 folds':        np.mean(test_acc_all),
        'average train time of 10 folds': np.mean(train_time_all),
        'average test time of 10 folds':  np.mean(test_time_all),
    }, index=['seedvig']).to_csv(result_path / 'hyperparam.csv', index=True)

    print(f"\n{'=' * 40}")
    print(f"10-fold average accuracy:   {np.mean(test_acc_all):.4f}")
    print(f"10-fold average train time: {_fmt_time(np.mean(train_time_all))}")
    print(f"10-fold average test time:  {_fmt_time(np.mean(test_time_all))}")

    print(f"Results saved to: {result_path / 'summary.csv'}")


# eeg_eval - inference + per-class metrics
def _save_eval_results(stem, weight_path, lines, rows, accuracy):
    """Save inference report next to the weight file (.txt + .csv)."""
    out_dir = Path(weight_path).parent

    txt_path = out_dir / f'{stem}_eval.txt'
    csv_path = out_dir / f'{stem}_eval.csv'

    with open(txt_path, 'w') as f:
        f.write(f'Weight : {weight_path}\n')
        f.write('\n'.join(lines) + '\n')

    for row in rows:
        row['accuracy'] = float('nan')
    rows.append({
        'class': 'overall', 'precision': float('nan'), 'recall': float('nan'),
        'f1': float('nan'), 'support': int(sum(r['support'] for r in rows)),
        'accuracy': round(float(accuracy), 4),
    })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f'\nResults saved → {txt_path}')
    print(f'             → {csv_path}')

def eeg_eval(args, config, device):
    """Run inference with a saved weight file, print per-class metrics, and save results."""
    dataset     = config['dataset'].upper()
    weight_path = Path(args.load_weights)
    if not weight_path.exists():
        raise FileNotFoundError(f'Weight file not found: {weight_path}')

    if dataset == 'DEAP':
        dim, num_class, data_np, labels_int = _load_deap_data(args, config)
        print(f"Dimension : {dim}  ({num_class} classes)")
        print(f"Samples   : {len(data_np)}")
        model = _build_model('DEAP', config, device, weight_path)
        preds = _batch_infer(model, data_np, device, args.batch_size)
        lines, rows, accuracy = _print_report(preds, labels_int, _DEAP_CLASS_NAMES[dim])
        stem = f'DEAP_{dim}'

    elif dataset == 'SEEDVIG':
        num_class, data_np, labels = _load_seedvig_data(args, config)
        class_names = _SEEDVIG_CLASS_NAMES.get(
            len(config.get('perclos_thresholds', [0.35, 0.70])),
            [f'Class_{i}' for i in range(num_class)],
        )
        print(f"Classes   : {class_names}")
        print(f"Samples   : {len(data_np)}")
        print(f"Class dist: {dict(zip(*np.unique(labels, return_counts=True)))}")
        model = _build_model('SEEDVIG', config, device, weight_path)
        preds = _batch_infer(model, data_np, device, args.batch_size)
        lines, rows, accuracy = _print_report(preds, labels, class_names)
        stem = 'SEEDVIG' + (f'_{args.session}' if args.session else '')

    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    _save_eval_results(stem, weight_path, lines, rows, accuracy)


# eeg_features - capsule-vector extraction
def eeg_features(args, config, device):
    dataset     = config['dataset'].upper()
    weight_path = Path(args.load_weights)
    if not weight_path.exists():
        raise FileNotFoundError(f'Weight file not found: {weight_path}')

    feat_dir = Path(args.work_dir) / 'features' if args.work_dir else Path('./features')
    feat_dir.mkdir(parents=True, exist_ok=True)

    if dataset == 'DEAP':
        dim, num_class, data_np, labels_int = _load_deap_data(args, config)
        print(f"Dimension : {dim}  ({num_class} classes)")
        print(f"Samples   : {len(data_np)}")
        model               = _build_model('DEAP', config, device, weight_path)
        all_vecs, all_lens  = _batch_extract(model, data_np, device, args.batch_size)
        out_file            = feat_dir / f'DEAP_{dim}_features.npy'
        np.save(out_file, {'features': all_vecs, 'lengths': all_lens, 'labels': labels_int})
        label_shape         = labels_int.shape

    elif dataset == 'SEEDVIG':
        num_class, data_np, labels = _load_seedvig_data(args, config)
        print(f"Samples   : {len(data_np)}")
        model               = _build_model('SEEDVIG', config, device, weight_path)
        all_vecs, all_lens  = _batch_extract(model, data_np, device, args.batch_size)
        out_file            = feat_dir / 'SEEDVIG_features.npy'
        np.save(out_file, {'features': all_vecs, 'lengths': all_lens, 'labels': labels})
        label_shape         = labels.shape

    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    print(f"\nFeatures saved -> {out_file}")
    print(f"  features : {all_vecs.shape}  (capsule vectors)")
    print(f"  lengths  : {all_lens.shape}  (class probabilities)")
    print(f"  labels   : {label_shape}")
