import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import random_split


def load_data_magic():
    data = pd.read_csv(
        "./demos/data/magic04.csv", names=["y_" + str(i) for i in range(10)] + ["group"]
    )

    data_g = data[data["group"] == "g"].drop(["group"], axis=1)
    data_h = data[data["group"] == "h"].drop(["group"], axis=1)

    data_g = np.array(data_g)
    data_h = np.array(data_h)

    return data_g, data_h


def load_train_test_data_magic(train_portion=2 / 3, group="h"):
    data_g, data_h = load_data_magic()

    indices_g = int(np.ceil(data_g.shape[0] * train_portion))
    data_g_train = data_g[:indices_g, :]
    data_g_test = data_g[indices_g:, :]

    indices_h = int(np.ceil(data_h.shape[0] * train_portion))
    data_h_train = data_h[:indices_h, :]
    data_h_test = data_h[indices_h:, :]

    complete_train_data = np.concatenate((data_g_train, data_h_train), axis=0)

    if group == "g":
        y_train = data_g_train
        y_test = data_g_test
    elif group == "h":
        y_train = data_h_train
        y_test = data_h_test

    # normalizing all columns by the complete data
    y_train_normed = (y_train - complete_train_data.min(axis=0)) / (
        complete_train_data.max(axis=0) - complete_train_data.min(axis=0)
    )
    y_train_normed = (y_train_normed - 0.5) * 2

    y_test_normed = (y_test - complete_train_data.min(axis=0)) / (
        complete_train_data.max(axis=0) - complete_train_data.min(axis=0)
    )
    y_test_normed = (y_test_normed - 0.5) * 2

    return y_train_normed, y_test_normed


def split_tensor_train_validation_sets(
    data, num_folds, validation_fold_index, random_state
):
    # Assume you have a tensor called 'data' of shape (num_samples, ...)

    # Define the number of folds and the index of the validation fold
    # num_folds = 5
    # validation_fold_index = 2

    # Calculate the size of each fold
    fold_size = len(data) // num_folds

    # Split the data into folds
    data_folds = random_split(
        data,
        [fold_size] * (num_folds - 1) + [len(data) - (num_folds - 1) * fold_size],
        generator=torch.Generator().manual_seed(random_state),
    )

    # Retrieve the validation set using indices
    validation_indices = data_folds[validation_fold_index].indices
    validation_set = data[validation_indices]

    # Create the training set by concatenating the remaining folds
    training_folds = [
        fold for idx, fold in enumerate(data_folds) if idx != validation_fold_index
    ]
    training_indices = torch.cat(
        [torch.tensor(fold.indices) for fold in training_folds], dim=0
    )
    training_set = data[training_indices]

    # test
    # training_set.size(0) + validation_set.size(0) == data.size(0)

    return training_set, validation_set


def load_magic_data(
    group="h",
    train_portion=2 / 3,
    data_dims=10,
    poly_span_abs=15,
    cross_validation_folds=5,
    validation_fold_index=4,
    split_random_state=25,
):
    (
        y_train_normed,
        y_test_normed,
    ) = load_train_test_data_magic(train_portion=train_portion, group=group)
    y_train_normed = y_train_normed[:, :data_dims]
    y_test_normed = y_test_normed[:, :data_dims]

    # Data is normed between [-1,1], need to bring it into the polynomial span
    y_train_normed = y_train_normed * poly_span_abs
    y_test_normed = y_test_normed * poly_span_abs

    y_train = torch.FloatTensor(y_train_normed)
    y_test = torch.FloatTensor(y_test_normed)

    y_train, y_validate = split_tensor_train_validation_sets(
        y_train,
        num_folds=cross_validation_folds,
        validation_fold_index=validation_fold_index,
        random_state=split_random_state,
    )

    return y_train, y_validate, y_test


def magic_classification_report(
    loaded_model_g,
    loaded_model_h,
    data_g_test_normed,
    data_h_test_normed,
    data_h,
    data_g,
    phi_g=False,
    phi_h=False,
    return_auc=False,
):
    # data_g_test_normed = torch.tensor(data_g_test_normed)
    # data_h_test_normed = torch.tensor(data_h_test_normed)

    fh_g = torch.exp(loaded_model_h.log_likelihood(data_g_test_normed)).detach()
    fh_h = torch.exp(loaded_model_h.log_likelihood(data_h_test_normed)).detach()
    fg_g = torch.exp(loaded_model_g.log_likelihood(data_g_test_normed)).detach()
    fg_h = torch.exp(loaded_model_g.log_likelihood(data_h_test_normed)).detach()

    likelihood_predictions = [fh_g, fh_h, fg_g, fg_h]
    names = ["fh_g", "fh_h", "fg_g", "fg_h"]

    for i in range(4):
        likelihood = likelihood_predictions[i]
        name = names[i]
        if likelihood.isnan().any():
            nan_index = (likelihood.isnan() == True).nonzero(as_tuple=True)
            print(name, "is NAN at:", nan_index)
            print("the observation is:", data_g_test_normed[nan_index])

        if likelihood.isinf().any():
            nan_index = (likelihood.isnan() == True).nonzero(as_tuple=True)
            print(name, "is INF at:", nan_index)
            print("the observation is:", data_g_test_normed[nan_index])

    if phi_g == False:
        phi_g = data_g.shape[0] / (data_h.shape[0] + data_g.shape[0])
        phi_h = 1 - phi_g

    bayes_prediction_g_g = phi_g * fg_g / (phi_g * fg_g + phi_h * fh_g)
    bayes_prediction_h_h = phi_h * fh_h / (phi_g * fg_h + phi_h * fh_h)

    if bayes_prediction_g_g.isnan().any():
        nan_index = (bayes_prediction_g_g.isnan() == True).nonzero(as_tuple=True)
        print("bayes_prediction_g_g is NAN at:", nan_index)
        bayes_prediction_g_g[bayes_prediction_g_g.isnan() == True] = phi_g
        print("Nan are replaced with prior probability phi_g", phi_g)

    if bayes_prediction_h_h.isnan().any():
        nan_index = (bayes_prediction_h_h.isnan() == True).nonzero(as_tuple=True)
        print("bayes_prediction_h_h is NAN at:", nan_index)
        bayes_prediction_h_h[bayes_prediction_h_h.isnan() == True] = phi_h
        print("Nan are replaced with prior probability phi_h", phi_h)

    # G=1, H=0
    y = np.concatenate(
        (
            np.array([1.0] * data_g_test_normed.size(0)),
            np.array([0.0] * data_h_test_normed.size(0)),
        ),
        axis=0,
    )
    y_pred_proba = np.concatenate(
        (np.array(bayes_prediction_g_g), np.array(1 - bayes_prediction_h_h)), axis=0
    )

    fpr, tpr, _ = roc_curve(y, y_pred_proba)

    table_roc_curve = pd.DataFrame({"fpr": fpr, "tpr": tpr})

    if return_auc == True:
        # Compute AUC-ROC
        auc_roc = roc_auc_score(y, y_pred_proba)
        return auc_roc

    return table_roc_curve
