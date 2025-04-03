import torch
from torch.utils.data import Dataset, DataLoader
from gtm.gtm_model.gtm import GTM

if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    class Gaussian2DDataset(Dataset):
        def __init__(self, num_samples=2000):
            self.mean = torch.tensor([1.0, -1.0, 0.0])  
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
    model = GTM(
                transformation_spline_range=list([[-15], [15]]), 
                degree_decorrelation=10,
                degree_transformations=10,
                num_decorr_layers=3,
                number_variables=3,
                device="cpu") # somehow the model with the vectorised version only works with 3 dimensions minimum due to the vectorisation (makes sense) TODO: need to solve that

    # hyperparameter tune the model
    study = model.hyperparameter_tune_penalties( 
                                      train_dataloader=dataloader, 
                                      validate_dataloader=dataloader, 
                                      penvalueridge = ["sample"],
                                      penfirstridge = ["sample"],
                                      pensecondridge = ["sample"],
                                      ctm_pensecondridge = ["sample"],
                                      lambda_penalty_params = ["sample"],
                                      train_covariates=False, 
                                      validate_covariates=False, 
                                      adaptive_lasso_weights_matrix = False,
                                      learning_rate=1, 
                                      iterations=20, 
                                      patience=5, 
                                      min_delta=1e-7, 
                                      optimizer='LBFGS', 
                                      lambda_penalty_mode="square", 
                                      objective_type="negloglik", 
                                      seperate_copula_training=False,
                                      max_batches_per_iter=10,
                                        tuning_mode="optuna",
                              cross_validation_folds=False,
                              random_state_KFold=42,
                              device=None,
                              pretrained_transformation_layer=False,
                              n_trials=6,
                              temp_folder=".", 
                              study_name=None)
    
    penalty_params=torch.FloatTensor([
                                    study.best_params["penvalueridge"],
                                    study.best_params["penfirstridge"],
                                    study.best_params["pensecondridge"],
                                    study.best_params["ctm_pensecondridge"]
                                      ])
    adaptive_lasso_weights_matrix = False
    lambda_penalty_params=torch.FloatTensor([study.best_params["lambda_penalty_params"]])
    
    # pretrain the marginal transformations
    model.pretrain_tranformation_layer(dataloader, iterations=10, max_batches_per_iter=10, penalty_params=penalty_params)
    
    # train the joint model
    model.__train__(train_dataloader=dataloader, validate_dataloader=dataloader, iterations=10, optimizer="LBFGS",
                    penalty_params=penalty_params, adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix, lambda_penalty_params=lambda_penalty_params, 
                    max_batches_per_iter=10)

    # save model
    torch.save(model, "model_state_dict.pth")
    
    # Create the same model structure
    model2 = GTM(
                transformation_spline_range=list([[-15], [15]]), 
                degree_decorrelation=10,
                degree_transformations=10,
                num_decorr_layers=3,
                number_variables=3,
                device="cpu") 

    # Load saved weights
    model2 = torch.load("./model_state_dict.pth")

    # evaluate the likelihood of the data
    #model.likelihood(samples) # N x D
    evaluation_data = next(iter(dataloader))
    log_likelihood_evaluation_data = model2.log_likelihood(evaluation_data)
    
    # compute local pseudo precision matrix
    precision_matrix_evaluation_data = model2.compute_precision_matrix(evaluation_data)
    
    # compute local pseudo conditional correlation matrix TODO: add a method
    # ...
    
    # compute IAE, KLD TODO: add a method
    # ...
 
    # approximate transformation layer inverse (required for synthetic sampling to work)
    model2.approximate_transformation_inverse()

    # sample synthetic data
    synth_samples = model2.sample(n_samples=10000)
    
    # check approximation for fun
    print(synth_samples.mean(0))
    print(synth_samples.T.cov())
    print(synth_samples.T.cov() @ torch.diag(1 / torch.diag(synth_samples.T.cov())))
        

