# Sample Copulas Package
import numpy as np
import pyvinecopulib as pv

# Other Stuff
import torch
from torch.utils.data import DataLoader, Dataset

####### Create Joe Dummy Dataset #######


def create_simulated_data_joe_copula():
    global Copula_2D_Dataset  # Needed for the class

    copula_pv = pv.Bicop(
        family=pv.BicopFamily.joe, parameters=np.array([[2.5]]), rotation=90
    )
    # Train
    N_train = 2000
    simulated_data_uniform_train = copula_pv.simulate(n=N_train)
    simulated_data_train = (
        torch.distributions.Normal(0, 1)
        .icdf(torch.tensor(simulated_data_uniform_train))
        .float()
    )
    # Validate
    N_validate = 2000
    simulated_data_uniform_validate = copula_pv.simulate(n=N_validate)
    simulated_data_validate = (
        torch.distributions.Normal(0, 1)
        .icdf(torch.tensor(simulated_data_uniform_validate))
        .float()
    )
    # Test
    N_test = 20000
    simulated_data_uniform_test = copula_pv.simulate(n=N_test)
    simulated_data_test = (
        torch.distributions.Normal(0, 1)
        .icdf(torch.tensor(simulated_data_uniform_test))
        .float()
    )

    loglik_copula = np.log(copula_pv.pdf(simulated_data_uniform_train))
    log_marginals = (
        torch.distributions.Normal(0, 1)
        .log_prob(torch.tensor(simulated_data_train))
        .sum(1)
    )
    loglik_true_train = torch.tensor(loglik_copula) + log_marginals
    loglik_copula = np.log(copula_pv.pdf(simulated_data_uniform_validate))
    log_marginals = (
        torch.distributions.Normal(0, 1)
        .log_prob(torch.tensor(simulated_data_validate))
        .sum(1)
    )
    loglik_true_validate = torch.tensor(loglik_copula) + log_marginals
    loglik_copula = np.log(copula_pv.pdf(simulated_data_uniform_test))
    log_marginals = (
        torch.distributions.Normal(0, 1)
        .log_prob(torch.tensor(simulated_data_test))
        .sum(1)
    )
    loglik_true_test = torch.tensor(loglik_copula) + log_marginals

    copula_pv_est = pv.Bicop(family=pv.BicopFamily.joe, rotation=90)
    copula_pv_est.fit(simulated_data_uniform_train)
    means = simulated_data_train.mean(0)
    vars = simulated_data_train.var(0)

    loglik_copula = np.log(copula_pv_est.pdf(simulated_data_uniform_train))
    log_marginals = (
        torch.distributions.Normal(means, vars)
        .log_prob(torch.tensor(simulated_data_train))
        .sum(1)
    )
    loglik_true_est_train = torch.tensor(loglik_copula) + log_marginals
    loglik_copula = np.log(copula_pv_est.pdf(simulated_data_uniform_validate))
    log_marginals = (
        torch.distributions.Normal(means, vars)
        .log_prob(torch.tensor(simulated_data_validate))
        .sum(1)
    )
    loglik_true_est_validate = torch.tensor(loglik_copula) + log_marginals
    loglik_copula = np.log(copula_pv_est.pdf(simulated_data_uniform_test))
    log_marginals = (
        torch.distributions.Normal(means, vars)
        .log_prob(torch.tensor(simulated_data_test))
        .sum(1)
    )
    loglik_true_est_test = torch.tensor(loglik_copula) + log_marginals

    # estimate the Multivariate Normal Distribution as Model
    mean_mvn_model = simulated_data_train.mean(
        0
    )  # 0 to do mean across dim 0 not globally
    cov_mvn_model = simulated_data_train.T.cov()
    mvn_model = torch.distributions.MultivariateNormal(
        loc=mean_mvn_model, covariance_matrix=cov_mvn_model
    )
    log_likelihood_train_gaussian = mvn_model.log_prob(simulated_data_train)
    log_likelihood_validate_gaussian = mvn_model.log_prob(simulated_data_validate)
    log_likelihood_test_gaussian = mvn_model.log_prob(simulated_data_test)

    class Copula_2D_Dataset(Dataset):
        def __init__(self, data):
            # First Dimension (N) needs to be the samples
            # Second Dimension (D) is the dimensionality of the data
            self.data = data

        def __len__(self):
            return self.data.size(0)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create dataset and DataLoader
    dataset_train = Copula_2D_Dataset(simulated_data_train)
    dataloader_train = DataLoader(dataset_train, batch_size=N_train)
    dataset_validate = Copula_2D_Dataset(simulated_data_validate)
    dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)

    return (
        dataloader_train,
        dataloader_validate,
        simulated_data_train,
        simulated_data_validate,
        simulated_data_test,
        loglik_true_train,
        loglik_true_validate,
        loglik_true_test,
        loglik_true_est_train,
        loglik_true_est_validate,
        loglik_true_est_test,
        log_likelihood_train_gaussian,
        log_likelihood_validate_gaussian,
        log_likelihood_test_gaussian,
    )
