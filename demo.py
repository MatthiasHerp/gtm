import torch
from torch.utils.data import Dataset, DataLoader
from gtm.gtm_model.gtm import GTM
from gtm.gtm_model.tm import TM


if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    
    ###### Example Dummy Dataset ######
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
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

    
    ###### 1. Data Driven manner to find the optimal degrees for the trransformation layer splines ######
    model = GTM(
                transformation_spline_range=list([[-10], [10]]), 
                degree_decorrelation=10,
                degree_transformations=10,
                num_decorr_layers=3,
                num_trans_layers=1,
                number_variables=3,
                calc_method_bspline="deBoor",
                device="cpu") 
    # somehow the model with the vectorised version only works with 3 dimensions minimum due to the vectorisation (makes sense) TODO: need to solve that
    
    varying_degree_transformation = True
    if varying_degree_transformation == True:
        optimal_degrees_transformation, optimal_degrees_transformation_pvalues = model.find_minimal_transformation_degrees(dataloader, 
                                                                                                                        dataloader, 
                                                                                                                        iterations=100, 
                                                                                                                        degrees_try_list=list(range(5,155,5)),
                                                                                                                        max_batches_per_iter=False)
        
        model = GTM(
                    transformation_spline_range=list([[-10], [10]]), 
                    degree_decorrelation=10,
                    degree_transformations=optimal_degrees_transformation,
                    num_decorr_layers=3,
                    num_trans_layers=1,
                    number_variables=3,
                    calc_method_bspline="deBoor",
                    device="cpu") 
    else:
        optimal_degrees_transformation = model.degree_transformations
    

    ###### 2. Hyperparameter Tune the models penalties ######
    hyperparameter_tune = True
    if hyperparameter_tune == True:
        study = model.hyperparameter_tune_penalties( 
                                        train_dataloader=dataloader, 
                                        validate_dataloader=dataloader, 
                                        penvalueridge = [0], #["sample"],
                                        penfirstridge = ["sample"],
                                        pensecondridge = ["sample"],
                                        ctm_pensecondridge = ["sample"],
                                        lambda_penalty_params = [0], #["sample"],
                                        train_covariates=False, 
                                        validate_covariates=False, 
                                        adaptive_lasso_weights_matrix = False,
                                        learning_rate=1, 
                                        iterations=100, 
                                        patience=5, 
                                        min_delta=1e-7, 
                                        optimizer='LBFGS', 
                                        lambda_penalty_mode="square", 
                                        objective_type="negloglik", 
                                        seperate_copula_training=False,
                                        max_batches_per_iter=False,
                                        tuning_mode="optuna",
                                cross_validation_folds=False,
                                random_state_KFold=42,
                                device=None,
                                pretrained_transformation_layer=False,
                                n_trials=10,
                                temp_folder=".", 
                                study_name=None)
    
        penalty_params=torch.FloatTensor([
                                    0, #study.best_params["penvalueridge"],
                                    study.best_params["penfirstridge"],
                                    study.best_params["pensecondridge"],
                                    study.best_params["ctm_pensecondridge"]
                                      ])
        adaptive_lasso_weights_matrix = False
        lambda_penalty_params=False #torch.FloatTensor([study.best_params["lambda_penalty_params"]])
    else:
        penalty_params=torch.FloatTensor([
                                    0, 0, 0, 0
                                      ])
        adaptive_lasso_weights_matrix = False
        lambda_penalty_params=False
    
    ###### 3. Train the optimal model ######
    
    # pretrain the marginal transformations
    model.pretrain_tranformation_layer(dataloader, iterations=100, max_batches_per_iter=False, penalty_params=penalty_params)
    
    # train the joint model
    model.__train__(train_dataloader=dataloader, validate_dataloader=dataloader, iterations=100, optimizer="LBFGS",
                    penalty_params=penalty_params, adaptive_lasso_weights_matrix=adaptive_lasso_weights_matrix, lambda_penalty_params=lambda_penalty_params, 
                    max_batches_per_iter=False)
    
    ###### 4. How to save and load the trained model ######

    # save model
    torch.save(model, "model_state_dict.pth")
    
    # Create the same model structure
    model2 = GTM(
                transformation_spline_range=list([[-10], [10]]), 
                degree_decorrelation=10,
                degree_transformations=optimal_degrees_transformation,
                num_decorr_layers=3,
                num_trans_layers=1,
                number_variables=3,
                calc_method_bspline="deBoor",
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
    print(synth_samples.mean(0).numpy().round(3))
    print(synth_samples.T.cov().numpy().round(3))
    print((synth_samples.T.cov() @ torch.diag(1 / torch.diag(synth_samples.T.cov()))).numpy().round(3))
        

