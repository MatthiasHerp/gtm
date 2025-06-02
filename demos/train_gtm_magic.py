from gtm import *
from demos.load_magic_data_helpers import *
from demos.dataset_helpers import Generic_Dataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    
    num_decorr_layers = 3
    group="h"
    
    for depth in [3, 4, 5, 6, 7, 8, 9]:
        for group in ["h", "g"]:
                print("Running with depth:", depth, "group:", group, "num_decorr_layers:", num_decorr_layers)
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if group == "h":
                    variable_degree_list = [60, 60, 60, 60, 45, 105, 105, 60, 60, 75]
                elif group == "g":
                    variable_degree_list = [150, 150, 30, 45, 60, 60, 45, 60, 75, 30]

                poly_span_abs= 15
                
                y_train, y_validate, y_test = load_magic_data(group=group,
                                                            train_portion=2/3,
                                                            data_dims=10,
                                                            poly_span_abs=poly_span_abs,
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
                            transformation_spline_range=list([[-poly_span_abs], [poly_span_abs]]), 
                            decorrelation_spline_range=list([[-poly_span_abs], [poly_span_abs]]), 
                            degree_decorrelation=[40],
                            degree_transformations=[variable_degree_list],
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
                                            penvalueridge = [0],
                                            penfirstridge = ["sample"],
                                            pensecondridge = ["sample"],
                                            ctm_pensecondridge = ["sample"],
                                            lambda_penalty_params = [0],
                                            adaptive_lasso_weights_matrix=False,
                                            iterations=1000, 
                                            patience=5, 
                                            min_delta=1e-7, 
                                            optimizer='LBFGS', 
                                            tuning_mode="optuna",
                                            device=device,
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
                _ = model.__train__(train_dataloader=dataloader_train, validate_dataloader=dataloader_validate, 
                                    iterations=1000, optimizer="LBFGS",
                                    penalty_params=penalty_params, max_batches_per_iter=False)
                
                torch.save(model, 
                        "magic_" + "group_" 
                        + group + "decorr_" 
                        + num_decorr_layers + "_gtm_state_dict.pth")