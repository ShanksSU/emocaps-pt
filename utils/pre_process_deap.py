import os
import h5py
import numpy as np
import os.path as osp
import _pickle as cPickle

import torch
from torch.utils.data import TensorDataset, DataLoader


DEAP_TRAIN_IMAGE_COUNT = 2500


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DataDEAP:
    def __init__(self, args):
        self.args = dotdict(args)
        self.data_path = self.args.data_path
        self.label_type = self.args.label_type

    def pre_process(self, num_subject):
        for sub in range(num_subject):
            data_, label_ = self.load_data_per_subject(sub)
            data_rnn, data_cnn, label_ = self.apply_mixup(data_, label_)
            print(label_.shape)
            label_ = self.label_selection(label_)
            print('Data and label prepared!')
            print('data_rnn:' + str(data_rnn.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_rnn, data_cnn, label_, sub)

    def load_data_per_subject(self, sub):
        sub += 1
        sub_code = f's0{sub}.dat' if sub < 10 else f's{sub}.dat'
        subject_path = osp.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['labels']
        data = subject['data'][:, 0:32, :]
        channels = list(range(32))
        data = np.stack([data[:, c, :] for c in channels], axis=1)
        print(f'Processing subject: {sub}')
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label

    def label_selection(self, label):
        binary_label = label[:, ] > 5

        hv = label[:, 0] > 5
        lv = ~hv
        ha = label[:, 1] > 5
        la = ~ha

        hvha = np.bitwise_and(hv, ha)
        hvla = np.bitwise_and(hv, la)
        lvha = np.bitwise_and(lv, ha)
        lvla = np.bitwise_and(lv, la)

        va_label = np.empty(label.shape[0])
        va_label[hvha] = 0
        va_label[hvla] = 1
        va_label[lvha] = 2
        va_label[lvla] = 3
        va_label = np.expand_dims(va_label, axis=1)

        hd = label[:, 2] > 5
        ld = ~hd

        vad_label = np.empty(label.shape[0])
        vad_label[np.bitwise_and(hvha, hd)] = 0
        vad_label[np.bitwise_and(hvha, ld)] = 1
        vad_label[np.bitwise_and(hvla, hd)] = 2
        vad_label[np.bitwise_and(hvla, ld)] = 3
        vad_label[np.bitwise_and(lvha, hd)] = 4
        vad_label[np.bitwise_and(lvha, ld)] = 5
        vad_label[np.bitwise_and(lvla, hd)] = 6
        vad_label[np.bitwise_and(lvla, ld)] = 7
        vad_label = np.expand_dims(vad_label, axis=1)

        label = np.append(binary_label, va_label, axis=1)
        label = np.append(label, vad_label, axis=1)
        return label

    def save(self, data_rnn, data_cnn, label, sub):
        save_path = osp.join(self.args.preprocessed_path, 'data_' + self.args.data_format)
        os.makedirs(save_path, exist_ok=True)
        name = 'sub' + str(sub) + '.hdf'
        dataset = h5py.File(osp.join(save_path, name), 'w')
        dataset['data_rnn'] = data_rnn
        dataset['data_cnn'] = data_cnn
        dataset['label'] = label
        dataset.close()

    def load_per_subject(self, sub):
        save_path = self.args.preprocessed_path
        sub_code = 'sub' + str(sub) + '.hdf'
        path = osp.join(save_path, 'data_' + self.args.data_format, sub_code)
        dataset = h5py.File(path, 'r')
        data_rnn = np.array(dataset['data_rnn'])
        data_cnn = np.array(dataset['data_cnn'])
        label = np.array(dataset['label'])
        return data_rnn, data_cnn, label

    def apply_mixup(self, data, labels):
        data_in = data.transpose(0, 2, 1)
        window_size = self.args.window_size
        label_in = labels

        label_inter = np.empty([0, 4])
        data_inter_cnn = np.empty([0, window_size, 9, 9])
        data_inter_rnn = np.empty([0, window_size, 32])

        trials = data_in.shape[0]
        for trial in range(trials):
            base_signal = (
                data_in[trial, 0:128, 0:32] +
                data_in[trial, 128:256, 0:32] +
                data_in[trial, 256:384, 0:32]
            ) / 3

            data = data_in[trial, 384:, 0:32]
            for i in range(60):
                data[i * 128:(i + 1) * 128, 0:32] -= base_signal

            data = norm_dataset(data)
            data, label = segment_signal_without_transition(data, label_in, trial, window_size)

            data_cnn = dataset_1Dto2D(data)
            data_cnn = data_cnn.reshape(int(data_cnn.shape[0] / window_size), window_size, 9, 9)
            data_rnn = data.reshape(int(data.shape[0] / window_size), window_size, 32)

            data_inter_cnn = np.vstack([data_inter_cnn, data_cnn])
            data_inter_rnn = np.vstack([data_inter_rnn, data_rnn])
            label_inter = np.vstack([label_inter, label])

        return data_inter_rnn, data_inter_cnn, label_inter


def pre_process(data, label, num_classes=None):
    """Add channel dim and convert labels to one-hot."""
    data_np = data[..., None].astype('float32')
    label_onehot = np.eye(num_classes)[label.astype(int)].astype('float32')
    return data_np, label_onehot


def generate_data_loaders(X_train, y_train, X_test, y_test, batch_size):
    """Create PyTorch DataLoaders from numpy arrays."""
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0,        0,        0,        data[0],  0,        data[16], 0,        0,        0      )
    data_2D[1] = (0,        0,        0,        data[1],  0,        data[17], 0,        0,        0      )
    data_2D[2] = (data[3],  0,        data[2],  0,        data[18], 0,        data[19], 0,        data[20])
    data_2D[3] = (0,        data[4],  0,        data[5],  0,        data[22], 0,        data[21], 0      )
    data_2D[4] = (data[7],  0,        data[6],  0,        data[23], 0,        data[24], 0,        data[25])
    data_2D[5] = (0,        data[8],  0,        data[9],  0,        data[27], 0,        data[26], 0      )
    data_2D[6] = (data[11], 0,        data[10], 0,        data[15], 0,        data[28], 0,        data[29])
    data_2D[7] = (0,        0,        0,        data[12], 0,        data[30], 0,        0,        0      )
    data_2D[8] = (0,        0,        0,        data[13], data[14], data[31], 0,        0,        0      )
    return data_2D


def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    return norm_dataset_1D


def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data.copy()
    data_normalized[data_normalized.nonzero()] = (
        data_normalized[data_normalized.nonzero()] - mean
    ) / sigma
    return data_normalized


def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    return dataset_2D


def windows(data, size):
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += size


def segment_signal_without_transition(data, label, label_index, window_size):
    for (start, end) in windows(data, window_size):
        if len(data[start:end]) == window_size:
            if start == 0:
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])
                labels = np.array(label[label_index, :])
                labels = np.vstack([labels, np.array(label[label_index, :])])
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.vstack([labels, np.array(label[label_index, :])])
    return segments, labels
