import torch
from torch.utils.data import Dataset, DataLoader
import math


def create_simulated_data_gmm():
    torch.manual_seed(0)

    N_train = 2000
    N_validate = 2000
    N_test = 20000

    def sample_gmm(n):
        component = torch.bernoulli(torch.full((n,), 0.5))
        means = torch.where(component == 0, -1.0, 2.0)
        stds = torch.where(component == 0, math.sqrt(0.5), math.sqrt(2.0))
        return torch.normal(means, stds).unsqueeze(1)

    def compute_log_likelihood(x):
        norm0 = torch.distributions.Normal(-1.0, math.sqrt(0.5))
        norm1 = torch.distributions.Normal(2.0, math.sqrt(2.0))
        p0 = torch.exp(norm0.log_prob(x))
        p1 = torch.exp(norm1.log_prob(x))
        return torch.log(0.5 * (p0 + p1)).squeeze()

    simulated_data_train = sample_gmm(N_train)
    simulated_data_validate = sample_gmm(N_validate)
    simulated_data_test = sample_gmm(N_test)

    loglik_train = compute_log_likelihood(simulated_data_train)
    loglik_validate = compute_log_likelihood(simulated_data_validate)
    loglik_test = compute_log_likelihood(simulated_data_test)

    class GMM_1D_Dataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return self.data.size(0)

        def __getitem__(self, idx):
            return self.data[idx]

    dataloader_train = DataLoader(
        GMM_1D_Dataset(simulated_data_train), batch_size=N_train
    )
    dataloader_validate = DataLoader(
        GMM_1D_Dataset(simulated_data_validate), batch_size=N_validate
    )

    return (
        dataloader_train,
        dataloader_validate,
        simulated_data_train,
        simulated_data_validate,
        simulated_data_test,
        loglik_train,
        loglik_validate,
        loglik_test,
    )
