####################################################################################################
####################################### Training GTMs ##############################################
# In this script we train GTMs for both the Hadron and Gamma classes of the MAGIC dataset.
# The GTMs are trained with different numbers of decorrelation layers, and the best penalites
# are selected using a validation set.
# Hyperparameter tuning uses the optuna package to generate proposed penalites.
# The GTM marginal degrees are based on the heuristic on demo_magic_marginal_transformations_degrees.ipynb
# The resulting models are analyzed in demo_magic_classification.ipynb.
# As the dataset is quite large we do not use a lasso penalty towards conditional independence
# Due to flexible marginals we use a marginal curvature penalty
####################################################################################################

from gtm import *
from demos.load_analyze_magic_data_helpers import *
from demos.dataset_helpers import Generic_Dataset
from torch.utils.data import DataLoader
import os

if __name__ == "__main__":
    
    for num_decorr_layers in [3, 4, 5, 6, 7, 8, 9]:
        for group in ["h", "g"]:
                print("Running for group:", group, " with num_decorr_layers:", num_decorr_layers)
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if group == "h":
                    variable_degree_list = [20, 35, 10, 10, 5, 30, 30, 40, 40, 10]
                elif group == "g":
                    variable_degree_list = [145, 150, 15, 5, 10, 30, 55, 60, 150, 10]
                
                y_train, y_validate, y_test = load_magic_data(group=group,
                                                            train_portion=2/3,
                                                            data_dims=10,
                                                            poly_span_abs=12,
                                                            cross_validation_folds=5,
                                                            validation_fold_index=4,
                                                            split_random_state=25)
                N_train = y_train.size(0)
                N_validate = y_validate.size(0)
                    
                # Create dataset and DataLoader
                dataset_train = Generic_Dataset(y_train)
                dataloader_train = DataLoader(dataset_train, batch_size=N_train)

                dataset_validate = Generic_Dataset(y_validate)
                dataloader_validate = DataLoader(dataset_validate, batch_size=N_validate)
                
                model = GTM(
                            transformation_spline_range=list([[-15], [15]]), 
                            degree_decorrelation=40,
                            degree_transformations=variable_degree_list,
                            num_decorr_layers=num_decorr_layers,
                            num_trans_layers=1,
                            number_variables=10,
                            spline_decorrelation="bspline",
                            spline_transformation="bspline",
                            device=device) 
                
                model.to(device)                       
                
                
                study = model.hyperparameter_tune_penalties( 
                                            train_dataloader=dataloader_train, 
                                            validate_dataloader=dataloader_validate, 
                                            penvalueridge = 0,
                                            penfirstridge = "sample",
                                            pensecondridge = "sample",
                                            ctm_pensecondridge = "sample",
                                            lambda_penalty_params = 0,
                                            adaptive_lasso_weights_matrix=False,
                                            iterations=1000, 
                                            patience=20, 
                                            min_delta=1e-8, 
                                            optimizer='LBFGS', 
                                            pretrained_transformation_layer=True,
                                            n_trials=40)
                
                penalty_params=torch.FloatTensor([
                                        0, #study.best_params["penvalueridge"],
                                        study.best_params["penfirstridge"],
                                        study.best_params["pensecondridge"],
                                        study.best_params["ctm_pensecondridge"]
                                        ])

                # pretrain the marginal transformations
                _ = model.pretrain_tranformation_layer(dataloader_train, iterations=1000)
                
                # train the joint model
                _ = model.train(train_dataloader=dataloader_train, validate_dataloader=dataloader_validate, 
                                    iterations=1000, optimizer="LBFGS",
                                    penalty_params=penalty_params)
                
                torch.save(model, os.path.join("demos", "models", f"magic_group_{group}_decorr_{num_decorr_layers}_gtm_state_dict.pth"))