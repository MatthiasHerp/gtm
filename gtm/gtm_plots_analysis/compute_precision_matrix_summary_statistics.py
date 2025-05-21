import torch
import pandas as pd

def compute_precision_matrix_summary_statistics(precision_matrix_train):
    """

    :param precision_matrix_train: torch.tensor
    :return:
    """
    data_dims = precision_matrix_train.size(1)

    # pd.DataFrame(precision_matrix_train).describe()
    precision_matrix_train = torch.tril(precision_matrix_train, diagonal=-1)

    precision_matrix_summary_statistics = pd.DataFrame({
        "abs_mean": torch.abs(precision_matrix_train).mean(0).numpy().flatten(),
        "mean": precision_matrix_train.mean(0).numpy().flatten(),
        "std": precision_matrix_train.std(0).numpy().flatten(),
        "min": precision_matrix_train.min(0).values.numpy().flatten(),
        "q_1": precision_matrix_train.quantile(q=0.01, dim=0).numpy().flatten(),
        "q_5": precision_matrix_train.quantile(q=0.05, dim=0).numpy().flatten(),
        "q_10": precision_matrix_train.quantile(q=0.1, dim=0).numpy().flatten(),
        "q_25": precision_matrix_train.quantile(q=0.25, dim=0).numpy().flatten(),
        "q_50": precision_matrix_train.quantile(q=0.5, dim=0).numpy().flatten(),
        "q_75": precision_matrix_train.quantile(q=0.75, dim=0).numpy().flatten(),
        "q_90": precision_matrix_train.quantile(q=0.9, dim=0).numpy().flatten(),
        "q_95": precision_matrix_train.quantile(q=0.95, dim=0).numpy().flatten(),
        "q_99": precision_matrix_train.quantile(q=0.99, dim=0).numpy().flatten(),
        "max": precision_matrix_train.max(0).values.numpy().flatten()
    })
    #
    precision_matrix_summary_statistics = precision_matrix_summary_statistics.loc[
        ~(precision_matrix_summary_statistics == 0).all(axis=1)]
    #
    precision_matrix_summary_statistics["var_row"] = torch.tril_indices(data_dims, data_dims, offset=-1)[0,
                                                     :].numpy().flatten()
    precision_matrix_summary_statistics["var_col"] = torch.tril_indices(data_dims, data_dims, offset=-1)[1,
                                                     :].numpy().flatten()
    #
    precision_matrix_summary_statistics = precision_matrix_summary_statistics[
        ['var_row', 'var_col', "abs_mean", "mean", "std", "min", "q_1", "q_5", "q_10", "q_25", "q_50", "q_75", "q_90", "q_95", "q_99", "max"]]

    return precision_matrix_summary_statistics