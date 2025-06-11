from gtm import *
from demos.load_analyze_magic_data_helpers import *
import os

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    group = "h"
    num_decorr_layers = 8
    loaded_model_h = torch.load(
        os.path.join(
            "demos",
            "models",
            f"magic_group_{group}_decorr_{num_decorr_layers}_gtm_state_dict.pth",
        ),
        map_location=torch.device(device),
    )
    loaded_model_h.device = device

    loaded_model_h.approximate_transformation_inverse()

    y_train_group_h, _, _ = load_magic_data(
        group=group,
        train_portion=2 / 3,
        data_dims=10,
        poly_span_abs=12,
        cross_validation_folds=5,
        validation_fold_index=4,
        split_random_state=25,
    )
    y_train_group_h = y_train_group_h.to(device)

    # Path to where the table will be stored
    table_path = os.path.join(
        "demos", "ci_tables", "conditional_independence_table_group_h.csv"
    )

    # set bounds at bounds of training data
    z_tilde = loaded_model_h.after_transformation(y_train_group_h)
    lower = z_tilde.min()
    upper = z_tilde.max()
    lower = (lower - 0.05 * (upper - lower)).item()
    upper = (upper + 0.05 * (upper - lower)).item()

    # Compute the table
    conditional_independence_table_h = (
        loaded_model_h.compute_conditional_independence_table(
            y=None,
            x=False,
            evaluation_data_type="samples_from_model",
            num_processes=1,
            sample_size=10000,
            num_points_quad=20,
            optimized=True,
            copula_only=True,
            min_val=lower,
            max_val=upper,
        )
    )

    # Save the table
    conditional_independence_table_h.to_csv(table_path)

    group = "g"
    num_decorr_layers = 6
    loaded_model_g = torch.load(
        os.path.join(
            "demos",
            "models",
            f"magic_group_{group}_decorr_{num_decorr_layers}_gtm_state_dict.pth",
        ),
        map_location=torch.device(device),
    )
    loaded_model_g.device = device

    loaded_model_g.approximate_transformation_inverse()

    y_train_group_g, _, _ = load_magic_data(
        group=group,
        train_portion=2 / 3,
        data_dims=10,
        poly_span_abs=12,
        cross_validation_folds=5,
        validation_fold_index=4,
        split_random_state=25,
    )
    y_train_group_g = y_train_group_g.to(device)

    # Path to where the table will be stored
    table_path = os.path.join(
        "demos", "ci_tables", "conditional_independence_table_group_g.csv"
    )

    # set bounds at bounds of training data
    z_tilde = loaded_model_g.after_transformation(y_train_group_g)
    lower = z_tilde.min()
    upper = z_tilde.max()
    lower = (lower - 0.05 * (upper - lower)).item()
    upper = (upper + 0.05 * (upper - lower)).item()

    # Compute the table
    conditional_independence_table_g = (
        loaded_model_g.compute_conditional_independence_table(
            y=None,
            x=False,
            evaluation_data_type="samples_from_model",
            num_processes=1,
            sample_size=10000,
            num_points_quad=20,
            optimized=True,
            copula_only=True,
            min_val=lower,
            max_val=upper,
        )
    )

    # Save the table
    conditional_independence_table_g.to_csv(table_path)
