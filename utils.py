import numpy as np
import pandas as pd

import pywt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def parse_dataset(path):
    data = pd.read_csv(path, sep='\t')
    features = data.select_dtypes(include=['float64'])
    labels = data.select_dtypes(include=['int64'])
    return features, labels

def preprocess_data(data, smoothing_window=1, downsample_window=1):
    processed_data = data.rolling(smoothing_window, axis=1).mean()
    processed_data = processed_data.dropna(axis=1)
    processed_data = processed_data[processed_data.columns[::downsample_window]]
    return processed_data

def label_encoder(labels):
    labels["1"] = labels["1"].apply(lambda x: x - 1)
    return labels

def label_decoder(labels):
    labels = labels + 1
    return labels

def results_report(y_true, y_pred, plot_cm=True):
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    if plot_cm:
        # plt.figure(figsize=)
        sns.heatmap(cm, annot=True, linewidths=.5, cmap="Blues", fmt='g', xticklabels=range(1,6), yticklabels=range(1,6))

    return cm

def get_cwt_features(data, decomp_level=1, wavelet="db2"):
    cwt_features = []
    for i, row in data.iterrows():
        coeffs = pywt.wavedec(row, wavelet, level=decomp_level)
        nth_level_approx_coeffs = coeffs[0]
        cwt_features.append(nth_level_approx_coeffs)

    X = pd.DataFrame(cwt_features, index=data.index)

    return X

def create_datasets_deprc(X, y, test_size=0.2, time_dim_first=False):
    # enc = LabelEncoder()
    # y_enc = enc.fit_transform(y)
    y_enc = y
    X_grouped = create_grouped_array(X)
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)
    X_train, X_val, y_train, y_val = train_test_split(X_grouped, y_enc, test_size=test_size)
    X_train, X_val = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_val)]
    y_train, y_val = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_val)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_val, y_val)
    return train_ds, valid_ds

def create_datasets(X, y, cv=0, time_dim_first=False):

    X_grouped = create_grouped_array(X)
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)

    train_folds = []
    val_folds = []

    # no split
    if cv==0:
        X_train, X_val = [torch.tensor(arr, dtype=torch.float32) for arr in (X_grouped, X_grouped)]
        y_train, y_val = [torch.tensor(arr, dtype=torch.long) for arr in (y, y)]
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_train, y_train)
        train_folds.append(train_ds)
        val_folds.append(val_ds)

    # do a simple train\test split
    elif cv==1:
        X_train, X_val, y_train, y_val = train_test_split(X_grouped, y, stratify=y, test_size=0.2)
        print(type(X_train), type(y_train))
        X_train, X_val = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_val)]
        y_train, y_val = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_val)]
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)

        train_folds.append(train_ds)
        val_folds.append(val_ds)

    # do a kfold split
    else:
        skf = StratifiedKFold(n_splits=cv)

        for train_index, val_index in skf.split(X_grouped, y):
            X_train, X_val = X_grouped[train_index], X_grouped[val_index]
            y_train, y_val = y[train_index], y[val_index]
            X_train, X_val = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_val)]
            y_train, y_val = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_val)]
            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)

            train_folds.append(train_ds)
            val_folds.append(val_ds)

    return train_folds, val_folds

def create_grouped_array(data, group_col='series_id'):
    X_grouped = np.row_stack([
        group.values[None]
        for _, group in data.groupby(data.index)])
    return X_grouped


def create_test_dataset(X):
    X_grouped = np.row_stack([
        group.values[None]
        for _, group in X.groupby(X.index)])
    X_grouped = torch.tensor(X_grouped.transpose(0, 2, 1)).float()
    y_fake = torch.tensor([0] * len(X_grouped)).long()
    return TensorDataset(X_grouped, y_fake)


def create_loaders(train_ds, valid_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()


class CyclicLR(_LRScheduler):

    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler

def plot_cosine_sched(n=100):
    sched = cosine(n)
    lrs = [sched(t, 1) for t in range(n * 4)]
    plt.plot(lrs)