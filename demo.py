import torch
from torch.utils.data import Dataset, DataLoader
from gtm.gtm_model.gtm import GTM

if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    class Gaussian2DDataset(Dataset):
        def __init__(self, num_samples=2000):
            self.mean = torch.tensor([0.0, 0.0, 0.0])  
            self.cov = torch.tensor([[1.0, 0.0, 0.4], 
                                     [0.0, 1.0, 0.4], 
                                     [0.4, 0.4, 1.0]])
            self.data = torch.distributions.MultivariateNormal(self.mean, covariance_matrix=self.cov).sample((num_samples,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]

    # Create dataset and DataLoader
    dataset = Gaussian2DDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # create GTM model
    model = GTM(input_min=-5, 
                input_max=5, 
                polynomial_range=list([[-5], [5]]),
                degree_decorrelation=10,
                degree_transformations=10,
                span_factor=torch.tensor(0.01), 
                span_restriction="reluler",
                num_decorr_layers=3,
                number_variables=3) # somehow the model with the vectorised version only works with 3 dimensions minimum due to the vectorisation (makes sense) TODO: need to solve that

    
    # pretrain the marginal transformations
    model.pretrain_tranformation_layer(dataloader, iterations=10)
    
    # TODO: next step to refactor
    # hyperparameter tune the model
    # model.hyperparameter_tune(y, n_epochs=1000)
    
    # train the joint model
    model.__train__(train_dataloader=dataloader, validate_dataloader=dataloader, iterations=10, optimizer="LBFGS",
                    penalty_params=torch.FloatTensor([0,0,0,0]), adaptive_lasso_weights_matrix = False, lambda_penalty_params=False)

    # evaluate the likelihood of the data
    #model.likelihood(samples) # N x D
    evaluation_data = next(iter(dataloader))
    log_likelihood_evaluation_data = model.log_likelihood(evaluation_data)
    
    # compute local pseudo precision matrix
    precision_matrix_evaluation_data = model.compute_precision_matrix(evaluation_data)
    
    # compute local pseudo conditional correlation matrix TODO: add a method
    # ...
    
    # compute IAE, KLD TODO: add a method
    # ...
 
    # approximate transformation layer inverse (required for synthetic sampling to work)
    model.approximate_transformation_inverse()

    # sample synthetic data
    synth_samples = model.sample(n_samples=10000)
    
    # check approximation for fun
    print(synth_samples.mean(0))
    print(synth_samples.T.cov())
    print(synth_samples.T.cov() @ torch.diag(1 / torch.diag(synth_samples.T.cov())))
        
