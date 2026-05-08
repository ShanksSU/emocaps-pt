import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import TensorDataset, DataLoader

TOTAL_CHANNELS = 17
CHANNEL_NAMES  = [
    'FT7', 'T7',  'TP7', 'CP5', 'CP3',
    'P5',  'P3',  'P1',  'PZ',  'P2',
    'P4',  'P6',  'PO7', 'PO3', 'POZ',
    'PO4', 'PO8',
]


def _resolve_channel_indices(channels_cfg):
    if channels_cfg is None:
        return list(range(TOTAL_CHANNELS))

    indices = []
    for ch in channels_cfg:
        if not (1 <= ch <= TOTAL_CHANNELS):
            raise ValueError(
                f"Channel {ch} is out of range. "
                f"Valid range: 1–{TOTAL_CHANNELS}."
            )
        indices.append(ch - 1)   # convert to 0-based
    return sorted(set(indices))


def discretize_perclos(perclos, thresholds):
    labels = np.zeros(len(perclos), dtype=np.int64)
    for i, t in enumerate(thresholds):
        labels[perclos.ravel() > t] = i + 1
    return labels


class DataSEEDVIG:
    def __init__(self, config):
        self.data_path  = config['data_path']
        self.de_key     = config.get('de_feature', 'de_movingAve')
        self.thresholds = config.get('perclos_thresholds', [0.35, 0.70])
        self.ch_indices = _resolve_channel_indices(config.get('channels'))

        # Update input_shape[0] so the model adapts to the channel count
        n_ch = len(self.ch_indices)
        config['input_shape'] = [n_ch, config['input_shape'][1], config['input_shape'][2]]

        # Human-readable info
        selected_names = [CHANNEL_NAMES[i] for i in self.ch_indices]
        if len(self.ch_indices) == TOTAL_CHANNELS:
            print(f"Channels: all {TOTAL_CHANNELS}  {selected_names}")
        else:
            nums = [i + 1 for i in self.ch_indices]
            print(f"Channels: {nums}  ->  {selected_names}")

        de_dir    = os.path.join(self.data_path, 'DE')
        label_dir = os.path.join(self.data_path, 'perclos_labels')

        filenames = sorted(f for f in os.listdir(de_dir) if f.endswith('.mat'))
        self.sessions  = [os.path.splitext(f)[0] for f in filenames]
        self.de_dir    = de_dir
        self.label_dir = label_dir

    def load_session(self, session_name):
        de_mat = sio.loadmat(os.path.join(self.de_dir,    session_name + '.mat'))
        lb_mat = sio.loadmat(os.path.join(self.label_dir, session_name + '.mat'))

        # [17, T, 5] -> select channels -> [T, n_ch, 5]
        raw = de_mat[self.de_key]                        # [17, T, 5]
        eeg = raw[self.ch_indices, :, :]                 # [n_ch, T, 5]
        eeg = eeg.transpose(1, 0, 2).astype(np.float32) # [T, n_ch, 5]

        perclos = lb_mat['perclos'].ravel().astype(np.float64)
        labels  = discretize_perclos(perclos, self.thresholds)
        return eeg, labels

    def load_all(self):
        all_eeg, all_labels = [], []
        for sess in self.sessions:
            eeg, labels = self.load_session(sess)
            all_eeg.append(eeg)
            all_labels.append(labels)
        return np.concatenate(all_eeg, axis=0), np.concatenate(all_labels, axis=0)


def pre_process_seedvig(eeg, labels, num_classes):
    eeg_out  = eeg[..., np.newaxis].astype(np.float32)
    label_oh = np.eye(num_classes)[labels].astype(np.float32)
    return eeg_out, label_oh


def generate_seedvig_loaders(X_train, y_train, X_test, y_test, batch_size):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader