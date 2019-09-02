import os

import numpy as np

def ls_files(path):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


def load_datasets(folder_path, ds_class, *args, **kwargs):
    ds_files = ls_files(folder_path)

    datasets = list()
    for ds_file in ds_files:
        datasets.append(ds_class(ds_file, *args, **kwargs))

    return datasets

def load_csvs(filenames, *args, **kwargs):
    all_csvs = list()
    for f in filenames:
        all_csvs.append(np.squeeze(np.loadtxt(f, *args, **kwargs)))

    return all_csvs

def load_csvs_in_path(folder_path, *args, **kwargs):
    csv_files = ls_files(folder_path)

    all_csvs = load_csvs(csv_files, *args, **kwargs)

    return all_csvs, csv_files

def load_labels(folder_path, label_idx):
    label_files = ls_files(folder_path)

    all_labels = list()
    for f in label_files:
        labels = np.squeeze(np.loadtxt(f)[:, label_idx])
        if len(labels) != 200:
            print(f)
        all_labels.append(labels)

    return all_labels

def random_split(arr, size, seed=None):
    np.random.seed(seed)
    val_idxs = np.random.choice(len(arr), size).tolist()

    first_set = [arr[i] for i in range(len(arr)) if i not in val_idxs]
    second_set = [arr[i] for i in range(len(arr)) if i in val_idxs]

    return first_set, second_set
