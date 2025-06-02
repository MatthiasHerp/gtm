import torch
import numpy as np
from torch.utils.data import random_split

import pandas as pd


def load_data_magic():
    data = pd.read_csv("./demos/data/magic04.csv", names=["y_" + str(i) for i in range(10)] + ["group"])

    data_g = data[data["group"] == "g"].drop(["group"], axis=1)
    data_h = data[data["group"] == "h"].drop(["group"], axis=1)

    data_g = np.array(data_g)
    data_h = np.array(data_h)

    return data_g, data_h


def load_train_test_data_magic(train_portion=2/3,
                               group="h"):

    data_g, data_h = load_data_magic()

    indices_g = int(np.ceil(data_g.shape[0] * train_portion))
    data_g_train = data_g[:indices_g, :]
    data_g_test = data_g[indices_g:, :]

    indices_h = int(np.ceil(data_h.shape[0] * train_portion))
    data_h_train = data_h[:indices_h, :]
    data_h_test = data_h[indices_h:, :]

    complete_train_data = np.concatenate((data_g_train,data_h_train),axis=0)

    if group == "g":
        y_train = data_g_train
        y_test = data_g_test
    elif group == "h":
        y_train = data_h_train
        y_test = data_h_test

    # normalizing all columns by the complete data
    y_train_normed = (y_train - complete_train_data.min(axis=0)) / (complete_train_data.max(axis=0) - complete_train_data.min(axis=0))
    y_train_normed = (y_train_normed - 0.5) * 2

    y_test_normed = (y_test - complete_train_data.min(axis=0)) / (complete_train_data.max(axis=0) - complete_train_data.min(axis=0))
    y_test_normed = (y_test_normed - 0.5) * 2

    return y_train_normed, y_test_normed


def split_tensor_train_validation_sets(data, num_folds, validation_fold_index,random_state):
    # Assume you have a tensor called 'data' of shape (num_samples, ...)

    # Define the number of folds and the index of the validation fold
    #num_folds = 5
    #validation_fold_index = 2

    # Calculate the size of each fold
    fold_size = len(data) // num_folds

    # Split the data into folds
    data_folds = random_split(data, [fold_size] * (num_folds - 1) + [len(data) - (num_folds - 1) * fold_size],
                              generator=torch.Generator().manual_seed(random_state))

    # Retrieve the validation set using indices
    validation_indices = data_folds[validation_fold_index].indices
    validation_set = data[validation_indices]

    # Create the training set by concatenating the remaining folds
    training_folds = [fold for idx, fold in enumerate(data_folds) if idx != validation_fold_index]
    training_indices = torch.cat([torch.tensor(fold.indices) for fold in training_folds], dim=0)
    training_set = data[training_indices]

    # test
    # training_set.size(0) + validation_set.size(0) == data.size(0)

    return training_set, validation_set


def load_magic_data(group="h",
                   train_portion=2/3,
                   data_dims=10,
                   poly_span_abs=15,
                   cross_validation_folds=5,
                   validation_fold_index=4,
                   split_random_state=25):

    y_train_normed, y_test_normed, = load_train_test_data_magic(train_portion=train_portion, group=group)
    y_train_normed = y_train_normed[:, :data_dims]
    y_test_normed = y_test_normed[:, :data_dims]

    # Data is normed between [-1,1], need to bring it into the polynomial span
    y_train_normed = y_train_normed * poly_span_abs
    y_test_normed = y_test_normed * poly_span_abs

    y_train = torch.FloatTensor(y_train_normed)
    y_test = torch.FloatTensor(y_test_normed)

    y_train, y_validate = split_tensor_train_validation_sets(y_train,
                                                             num_folds=cross_validation_folds,
                                                             validation_fold_index=validation_fold_index,
                                                             random_state=split_random_state)

    return y_train, y_validate, y_test

