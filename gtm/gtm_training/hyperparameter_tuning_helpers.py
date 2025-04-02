import torch
import itertools
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from python_nf_mctm.models.nf_mctm import *
from python_nf_mctm.training.training_helpers import *
from python_nf_mctm.training.training_helpers import train, if_float_create_lambda_penalisation_matrix
import optuna
from optuna.samplers import TPESampler

from python_nf_mctm.training.pretrain_transformation_layer import *
import multiprocessing as multi

from python_nf_mctm.models.nf_mctm import NF_MCTM
from python_nf_mctm.models.mctm import MCTM
from python_nf_mctm.models.factor_graph import Factor_Graph


# y_validate needed here as we pass it in the simulation study
def run_hyperparameter_tuning(y_train: torch.Tensor,
                              y_validate: torch.Tensor,
                              poly_span_abs: float,
                              iterations: int,
                              spline_decorrelation: str,
                              spline_transformation: str,
                              penvalueridge_list: list,
                              penfirstridge_list: list,
                              pensecondridge_list: list,
                              ctm_pensecondridge_list: list,
                              lambda_penalty_params_list: list,
                              learning_rate_list: list,
                              patience_list: list,
                              min_delta_list: list,
                              degree_transformations_list: list,
                              degree_decorrelation_list: list,
                              span_factor: torch.Tensor,
                              x_train: torch.Tensor = False,
                              x_validate: torch.Tensor = False,
                              tuning_mode="optuna",
                              cross_validation_folds=False,
                              tune_precision_matrix_penalty=False,
                              device=None,
                              pretrained_transformation_layer=False,
                              span_restriction="reluler",
                              n_trials=15,
                              initial_log_transform=False,
                              study_name=None,
                              total_num_threads=1,
                              covaraite_effect="multiplicative",
                              random_state_KFold=42,
                              linear_mctm=False,
                              list_comprehension=False,
                              num_decorr_layers=3,
                              num_trans_layers=1,
                              objective_type="negloglik",
                              calc_method_bspline="Naive_Basis",
                              optimizer="LBFGS",
                              lambda_penalty_mode="multiplicative",
                              match_after_preprocessing=False,
                              ema_decay=False,
                              factor_graph=False,
                              affine_decorr_layer=False,
                              degree_multi=False,
                              seperate_copula_training=False
                              ):
                          #normalisation_layer_list: list):
    """
    Generates List of all combinations of hyperparameter values from lists
    For each combination does a 5-fold cross validation and trains the model
    It then computes the log likelihood of the model on the validation set and
    stores it all in aa pandas dataframe

    grid search allows for lasso parameter hyperparameter tuning

    :param model:
    :param y:
    :param penvalueridge_list:
    :param penfirstridge_list:
    :param pensecondridge_list:
    :param learning_rate_list:
    :param patience_list:
    :param min_delta_list:
    :param degree_transformations_list:
    :param degree_decorrelation_list:
    :return:
    """

    list_of_lists = [penvalueridge_list, penfirstridge_list, pensecondridge_list, ctm_pensecondridge_list,
                     lambda_penalty_params_list,
                     learning_rate_list,
                     patience_list, min_delta_list,
                     degree_transformations_list, degree_decorrelation_list]
                     #normalisation_layer_list]
    hyperparameter_combinations_list = list(itertools.product(*list_of_lists))

    if x_train is False:
        number_covariates = 0
    else:
        number_covariates = 1

    if tuning_mode == "optuna":
        penvalueridge, penfirstridge, pensecondridge, ctm_pensecondridge, lambda_penalty_params, learning_rate, \
        patience, min_delta, degree_transformations, degree_decorrelation  = hyperparameter_combinations_list[0]
        
        # so model has no marginal part
        if seperate_copula_training == True:
            num_trans_layers = 0

        #if pretrained_transformation_layer == True:
            #transformation_model = pretrain_transformation_layer(y_train=y_train,
            #                                                     x_train=x_train,
            #                                                     number_covariates=number_covariates,
            #                                                     poly_span_abs=poly_span_abs,
            #                                                     degree_transformations=degree_transformations,
            #                                                     span_factor=span_factor,
            #                                                     monotonically_increasing=True,
            #                                                     spline=spline_transformation,
            #                                                     iterations=iterations,
            #                                                     patience=patience,
            #                                                     min_delta=min_delta,
            #                                                     return_report=False,
            #                                                     verbose=False,
            #                                                     initial_log_transform=initial_log_transform,
            #                                                     objective_type=objective_type,
            #                                                    pen_ctm_second_ridge=ctm_pensecondridge,
            #                                                    calc_method_bspline=calc_method_bspline)
            #
            #nf_mctm.pretrain_tranformation_layer(train_data=y_train,
            #    validate_data=y_validate,
            #    train_covariates=x_train,
            #    validate_covariates=x_validate,
            #    penalty_params=penalty_params,
            #    lambda_penalty_params=lambda_penalty_params,
            #    iterations=iterations, 
            #    learning_rate=learning_rate,
            #    patience=patience,
            #    min_delta=min_delta,
            #    verbose=False,
            #    optimizer=optimizer,
            #    lambda_penalty_mode=lambda_penalty_mode,
            #    objective_type=objective_type,
            #    return_plot=False,
            #    ema_decay=ema_decay)
        #    
        #    
        #else:
            #transformation_model = False

        def optuna_objective(trial, y_train=y_train, y_validate=y_validate): #, transformation_model=transformation_model): #,y_validate=y_validate

            # Defining the model
            poly_range = list([[-poly_span_abs], [poly_span_abs]]) #torch.FloatTensor

            #penvalueridge_opt  = trial.suggest_float("penvalueridge", 0.001, 5, log=True),

            #penvalueridge_opt = 0

            if tune_precision_matrix_penalty == True:
                #penfirstridge_opt = penfirstridge
                #pensecondridge_opt = pensecondridge
                penvalueridge_opt  = trial.suggest_float("penvalueridge", 0.0000001, 0.1, log=False) #log=True 0.1
                penfirstridge_opt =  trial.suggest_float("penfirstridge", 0.0000001, 0.1, log=False) # 5
                pensecondridge_opt = trial.suggest_float("pensecondridge", 0.0000001, 0.5, log=False) #0.0001 10
                ctm_pensecondridge_opt = 0 #trial.suggest_float("ctm_pensecondridge", 0.0001, 1, log=False) #0.0001

                lambda_penalty_params_opt = trial.suggest_float("lambda_penalty_params", 0.0000001, 0.1, log=False)

            else:
                penvalueridge_opt  = trial.suggest_float("penvalueridge", 0.0000001,   0.1, log=False) # 0.1
                penfirstridge_opt = trial.suggest_float("penfirstridge", 0.0000001, 0.1, log=False)
                pensecondridge_opt = trial.suggest_float("pensecondridge", 0.0000001, 0.5, log=False) #0.0001
                ctm_pensecondridge_opt = 0 #trial.suggest_float("ctm_pensecondridge", 0.0001, 1, log=False) #0.0001

                lambda_penalty_params_opt = lambda_penalty_params
                
            print("This Trial has the Hyperparameters:", 
                    "penvalueridge_opt:", penvalueridge_opt, " ", 
                    "penfirstridge_opt:", penfirstridge_opt, " ", 
                    "pensecondridge_opt:", pensecondridge_opt, " ", 
                    "ctm_pensecondridge_opt:", ctm_pensecondridge_opt, " ",
                    "lambda_penalty_params_opt:", lambda_penalty_params_opt)


            #lambda_penalty_params_opt = if_float_create_lambda_penalisation_matrix(lambda_penalty_params_opt, y_train) #TODO: uncomment this, trying if it solves factor graph issue
            penalty_params = torch.tensor([penvalueridge_opt,
                                           penfirstridge_opt,
                                           pensecondridge_opt,
                                           ctm_pensecondridge_opt])

            # for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(y.size()[0]))):
            #    y_train = y[train_idx, :]
            #    y_validate = y[val_idx, :]

            #Cross validation inspired by:
            #https://stackoverflow.com/questions/63224426/how-can-i-cross-validate-by-pytorch-and-optuna

            if cross_validation_folds == False:

                y_train = y_train.to(device)
                y_validate = y_validate.to(device)

                if linear_mctm:
                    nf_mctm = MCTM(input_min=y_train.min(0).values,  # y_sub_train
                                   input_max=y_train.max(0).values,  # y_sub_train
                                   polynomial_range=poly_range,
                                   number_variables=y_train.size()[1],  # y_sub_train
                                   spline_transformation=spline_transformation,
                                   spline_decorrelation=spline_decorrelation,
                                   degree_transformations=int(degree_transformations),
                                   degree_decorrelation=int(degree_decorrelation),
                                   span_factor=span_factor,  # device=device torch.tensor(
                                   span_restriction=span_restriction,
                                   number_covariates=number_covariates,
                                   list_comprehension=list_comprehension,
                                   num_decorr_layers=num_decorr_layers,
                                   num_trans_layers=num_trans_layers,
                                   initial_log_transform=initial_log_transform)  # device=device
                elif factor_graph:
                    nf_mctm = Factor_Graph( input_min = y_train.min(0).values, 
                                            input_max = y_train.max(0).values, 
                                            polynomial_range = list([[-5], [5]]),
                                            degree_transformations=int(degree_transformations),
                                            degree_decorrelation=int(degree_decorrelation),
                                            affine_decorr_layer=affine_decorr_layer, 
                                            degree_multi=int(degree_multi))
                else:
                    nf_mctm = NF_MCTM(input_min=y_train.min(0).values,  # y_sub_train
                                      input_max=y_train.max(0).values,  # y_sub_train
                                      polynomial_range=poly_range,
                                      number_variables=y_train.size()[1],  # y_sub_train
                                      spline_transformation=spline_transformation,
                                      spline_decorrelation=spline_decorrelation,
                                      degree_transformations=int(degree_transformations),
                                      degree_decorrelation=int(degree_decorrelation),
                                      span_factor=span_factor,  # device=device torch.tensor(
                                      span_restriction=span_restriction,
                                      number_covariates=number_covariates,
                                      list_comprehension=list_comprehension,
                                      num_decorr_layers=num_decorr_layers,
                                      num_trans_layers=num_trans_layers,
                                      initial_log_transform=initial_log_transform,
                                      calc_method_bspline=calc_method_bspline,
                                      affine_decorr_layer=affine_decorr_layer,
                                      degree_multi=degree_multi)  # device=device

                #nf_mctm = NF_MCTM(input_min=y_train.min(0).values,
                #                  input_max=y_train.max(0).values,
                #                  polynomial_range=poly_range,
                #                  number_variables=y_train.size(1),
                #                  spline_transformation=spline_transformation,
                #                  spline_decorrelation=spline_decorrelation,
                #                  degree_transformations=degree_transformations,
                #                  degree_decorrelation=degree_decorrelation,
                #                  number_covariates=number_covariates,
                #                  span_factor=span_factor,
                #                  span_restriction=span_restriction,
                #                  initial_log_transform=initial_log_transform,
                #                  covaraite_effect=covaraite_effect)
                ## normalisation_layer=normalisation_layer)

                if pretrained_transformation_layer == True:
                    
                    nf_mctm.pretrain_tranformation_layer(train_data=y_train,
                                                        validate_data=y_validate,
                                                        train_covariates=x_train,
                                                        validate_covariates=x_validate,
                                                        penalty_params=penalty_params,
                                                        lambda_penalty_params=lambda_penalty_params,
                                                        iterations=iterations, 
                                                        learning_rate=learning_rate,
                                                        patience=patience,
                                                        min_delta=min_delta,
                                                        verbose=False,
                                                        optimizer=optimizer,
                                                        lambda_penalty_mode=lambda_penalty_mode,
                                                        objective_type=objective_type,
                                                        return_plot=False,
                                                        ema_decay=ema_decay)
                    
                    #if seperate_copula_training == False:
                    #    nf_mctm.state_dict()["transformation.params"].data.copy_(transformation_model.state_dict()["l1.params"])
                    
                    #if seperate_copula_training == True:
                    #    y_train = transformation_model.forward(y_train,
                    #                                        covariate=False,
                    #                                        train=False,
                    #                                        compute_optimal_initial_params=False,
                    #                                        new_input=True).detach()
                    #    y_validate = transformation_model.forward(y_validate,
                    #                                        covariate=False,
                    #                                        train=False,
                    #                                        compute_optimal_initial_params=False,
                    #                                        new_input=True).detach()
                
                    #nf_mctm.state_dict()["transformation.params"].data.copy_(transformation_model.state_dict()["l1.params"])
                    
                    if match_after_preprocessing == True:
                        nf_mctm.transformation.params.requires_grad = False

                nf_mctm = nf_mctm.to(device)
                
                
                nf_mctm.train(train_data=y_train,
                            validate_data=y_validate, 
                            train_covariates=x_train,
                            validate_covariates=x_validate,
                            penalty_params=penalty_params,
                            lambda_penalty_params=lambda_penalty_params_opt,
                            iterations=iterations, 
                            learning_rate=learning_rate,
                            patience=patience,
                            min_delta=min_delta,
                            verbose=False,
                            optimizer=optimizer,
                            lambda_penalty_mode=lambda_penalty_mode,
                            objective_type=objective_type,
                            return_plot=False,
                            ema_decay=ema_decay)
                
                #train(model=nf_mctm,
                #      train_data=y_train,
                #      validate_data=y_validate,
                #      train_covariates=x_train,
                #      validate_covariates=x_validate,
                #      penalty_params=penalty_params,
                #      lambda_penalty_params=lambda_penalty_params_opt,
                #      iterations=iterations,
                #      learning_rate=learning_rate,
                #      patience=patience,
                #      min_delta=min_delta,
                #      verbose=False,
                #      return_report=False,
                #      objective_type=objective_type,
                #      lambda_penalty_mode=lambda_penalty_mode,
                #      optimizer=optimizer,
                #      ema_decay=ema_decay)  # no need for reporting and metrics,plots etc.

                if objective_type == "negloglik":
                    return nf_mctm.log_likelihood(y_validate, x_validate).cpu().detach().numpy().mean()#.mean() # we maximize loglikelihood
                elif objective_type == "score_matching":
                    return -1 * nf_mctm.exact_score_loss(y_validate, x_validate).cpu().detach().numpy().mean()#.mean() # -1 because we maximize reverse loss e.g. we minimize the loss
                elif objective_type == "noise_contrastive_estimation":
                    return nf_mctm.log_likelihood(y_validate, x_validate).cpu().detach().numpy().mean()  #TODO: For Now! Its cheating right?
            else:
                
                kf = KFold(n_splits=cross_validation_folds, shuffle=True, random_state=random_state_KFold)
                targets_list = []
                for train_idx, val_idx in kf.split(np.arange(y_train.size()[0])):
                    y_train_cv = y_train[train_idx, :]
                    y_validate_cv = y_train[val_idx, :] # TODO: changed from this... y_validate_cv[val_idx, :]

                    if x_train is False:
                        x_train_cv = False
                        x_validate_cv = False
                    else:
                        x_train_cv = x_train[train_idx]#, :
                        x_validate_cv = x_train[val_idx]#, :

                    y_train_cv = y_train_cv.to(device)
                    y_validate_cv = y_validate_cv.to(device)

                    if linear_mctm:
                        nf_mctm = MCTM(input_min=y_train.min(0).values,  # y_sub_train
                                       input_max=y_train.max(0).values,  # y_sub_train
                                       polynomial_range=poly_range,
                                       number_variables=y_train.size()[1],  # y_sub_train
                                       spline_transformation=spline_transformation,
                                       spline_decorrelation=spline_decorrelation,
                                       degree_transformations=int(degree_transformations),
                                       degree_decorrelation=int(degree_decorrelation),
                                       span_factor=span_factor,  # device=device torch.tensor(
                                       span_restriction=span_restriction,
                                       number_covariates=number_covariates,
                                       list_comprehension=list_comprehension,
                                       num_decorr_layers=num_decorr_layers,
                                       num_trans_layers=num_trans_layers,
                                       initial_log_transform=initial_log_transform)  # device=device
                    elif factor_graph:
                        nf_mctm = Factor_Graph( input_min = y_train.min(0).values, 
                                                input_max = y_train.max(0).values, 
                                                polynomial_range = list([[-5], [5]]),
                                                degree_transformations=int(degree_transformations),
                                                degree_decorrelation=int(degree_decorrelation),
                                                affine_decorr_layer=affine_decorr_layer, 
                                                degree_multi=int(degree_multi))
                    else:
                        nf_mctm = NF_MCTM(input_min=y_train.min(0).values,  # y_sub_train
                                          input_max=y_train.max(0).values,  # y_sub_train
                                          polynomial_range=poly_range,
                                          number_variables=y_train.size()[1],  # y_sub_train
                                          spline_transformation=spline_transformation,
                                          spline_decorrelation=spline_decorrelation,
                                          degree_transformations=int(degree_transformations),
                                          degree_decorrelation=int(degree_decorrelation),
                                          span_factor=span_factor,  # device=device torch.tensor(
                                          span_restriction=span_restriction,
                                          number_covariates=number_covariates,
                                          list_comprehension=list_comprehension,
                                          num_decorr_layers=num_decorr_layers,
                                          num_trans_layers=num_trans_layers,
                                          initial_log_transform=initial_log_transform,
                                          calc_method_bspline=calc_method_bspline,
                                          affine_decorr_layer=affine_decorr_layer,
                                          degree_multi=degree_multi)  # device=device

                    #nf_mctm = NF_MCTM(input_min=y_train_cv.min(0).values,
                    #                  input_max=y_train_cv.max(0).values,
                    #                  polynomial_range=poly_range,
                    #                  number_variables=y_train_cv.size()[1],
                    #                  spline_decorrelation=spline_decorrelation,
                    #                  degree_transformations=degree_transformations,
                    #                  degree_decorrelation=degree_decorrelation,
                    #                  number_covariates=number_covariates,
                    #                  span_factor=span_factor,
                    #                  span_restriction=span_restriction,
                    #                  initial_log_transform=initial_log_transform,
                    #                  covaraite_effect=covaraite_effect)
                    # normalisation_layer=normalisation_layer)

                    if pretrained_transformation_layer == True:
                        
                        nf_mctm.pretrain_tranformation_layer(train_data=y_train_cv,
                                    validate_data=y_validate_cv,
                                    train_covariates=x_train_cv,
                                    validate_covariates=x_validate_cv,
                                    penalty_params=penalty_params,
                                    lambda_penalty_params=lambda_penalty_params,
                                    iterations=iterations, 
                                    learning_rate=learning_rate,
                                    patience=patience,
                                    min_delta=min_delta,
                                    verbose=False,
                                    optimizer=optimizer,
                                    lambda_penalty_mode=lambda_penalty_mode,
                                    objective_type=objective_type,
                                    return_plot=False,
                                    ema_decay=ema_decay)
                        
                        
                        #if seperate_copula_training == False:
                        #    nf_mctm.state_dict()["transformation.params"].data.copy_(transformation_model.state_dict()["l1.params"])
                        #
                        #if seperate_copula_training == True:
                        #    y_train_cv = transformation_model.forward(y_train_cv,
                        #                                        covariate=False,
                        #                                        train=False,
                        #                                        compute_optimal_initial_params=False,
                        #                                        new_input=True).detach()
                        #    y_validate_cv = transformation_model.forward(y_validate_cv,
                        #                                        covariate=False,
                        #                                        train=False,
                        #                                        compute_optimal_initial_params=False,
                        #                                        new_input=True).detach()
                        #
                        #if match_after_preprocessing == True:
                        #    nf_mctm.transformation.params.requires_grad = False

                    nf_mctm = nf_mctm.to(device)

                    #print("nf_mctm",nf_mctm.device)
                    #print("y_train_cv", y_train_cv.device)
                    #print("y_validate_cv", y_validate_cv.device)
                    
                    nf_mctm.train(train_data=y_train_cv,
                            validate_data=y_validate_cv, 
                            train_covariates=x_train_cv,
                            validate_covariates=x_validate_cv,
                            penalty_params=penalty_params.to(device),
                            lambda_penalty_params=lambda_penalty_params_opt,
                            iterations=iterations, 
                            learning_rate=learning_rate,
                            patience=patience,
                            min_delta=min_delta,
                            verbose=False,
                            optimizer=optimizer,
                            lambda_penalty_mode=lambda_penalty_mode,
                            objective_type=objective_type,
                            return_plot=False,
                            ema_decay=ema_decay)

                    #train(model=nf_mctm,
                    #      train_data=y_train_cv,
                    #      validate_data=y_validate_cv,
                    #      train_covariates=x_train_cv,
                    #      validate_covariates=x_validate_cv,
                    #      penalty_params=penalty_params.to(device),
                    #      lambda_penalty_params=lambda_penalty_params_opt,
                    #      iterations=iterations,
                    #      learning_rate=learning_rate,
                    #      patience=patience,
                    #      min_delta=min_delta,
                    #      verbose=False,
                    #      return_report=False,
                    #      objective_type=objective_type,
                    #      lambda_penalty_mode=lambda_penalty_mode,
                    #      optimizer=optimizer,
                    #      ema_decay=ema_decay)  # no need for reporting and metrics,plots etc.

                    if objective_type == "negloglik":
                        target_cv = nf_mctm.log_likelihood(y_validate_cv, x_validate_cv).cpu().detach().numpy().mean()  # .mean()
                    elif objective_type == "score_matching":
                        target_cv = -1 * nf_mctm.exact_score_loss(y_validate_cv, x_validate_cv).cpu().detach().numpy().mean()  # .mean() # maximize the reverse score loss e.g.g minimize the loss
                    elif objective_type == "noise_contrastive_estimation":
                        target_cv = nf_mctm.log_likelihood(y_validate_cv, x_validate_cv).cpu().detach().numpy().mean()  #TODO: For Now! Its cheating right?
                    targets_list.append(target_cv)
                return np.mean(targets_list)

        # docs: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler
        
        study = optuna.create_study(sampler=TPESampler(n_startup_trials=int(np.floor(n_trials/2)), #7
                                                       consider_prior=False, #True # is this useful without a prior weight?
                                                       # Set consider_prior=False othwise with score matching we got the error: raise ValueError("Prior weight must be positive.")
                                                       prior_weight=0,#1.0, #default value 1.0 but then does not explore the space as good I think
                                                       multivariate=True # experimental but very useful here as our parameters are highly correlated
                                                       ),
                                    storage='sqlite:///hyperparameter_tuning_study.db',
                                    #hyperparameter_tuning_study.db',
                                    direction='maximize',
                                    study_name=study_name,
                                    load_if_exists=True) # we maximize the log likelihood
        
        #study = optuna.create_study(sampler=optuna.samplers.RandomSampler(),
        #                            storage='sqlite:///hyperparameter_tuning_study.db',
        #                            #hyperparameter_tuning_study.db',
        #                            direction='maximize',
        #                            study_name=study_name,
        #                            load_if_exists=True) # we maximize the log likelihood

        if total_num_threads > 1:

            from multiprocessing import Pool
            import tqdm

            sema= multi.Semaphore(total_num_threads)
            jobs = []
            for seed in range(1,21):
                sema.acquire()
                p = multi.Process(target=study.optimize, args=(optuna_objective, n_trials))
                jobs.append(p)
                p.start()

            for j in jobs:
                j.join()
        else:
            study.optimize(optuna_objective, n_trials=n_trials)

        print("hyperparameter_tuning done")
        return study

    elif tuning_mode == "grid":
        warnings.WarningMessage("Score loss with grid search not implemented. Only log likelihood based.")

        list_of_lists = [penvalueridge_list, penfirstridge_list, pensecondridge_list, ctm_pensecondridge_list,
                         lambda_penalty_params_list,
                         learning_rate_list,
                         patience_list, min_delta_list,
                         degree_transformations_list, degree_decorrelation_list]
                         #normalisation_layer_list]
        hyperparameter_combinations_list = list(itertools.product(*list_of_lists))

        #splits = KFold(n_splits=5, random_state=random_state_KFold) #want larger percent of data in train set? maybe another sampling method?
        # parallelisation even of the folds?

        results = pd.DataFrame(columns=['penvalueridge', 'penfirstridge', 'pensecondridge', 'ctm_pensecondridge', 'learning_rate',
                                        'patience', 'min_delta', 'degree_transformations',
                                        'degree_decorrelation', #'normalisation_layer',
                                        #'fold',
                                        'mean_validation_log_likelihood'])

        for hyperparameters in hyperparameter_combinations_list:
            penvalueridge, penfirstridge, pensecondridge, \
            lambda_penalty_param,\
            learning_rate, \
            patience, min_delta, \
            degree_transformations, degree_decorrelation  = hyperparameters #normalisation_layer

            # Defining the model
            poly_range = list([[-poly_span_abs], [poly_span_abs]]) #torch.FloatTensor
            penalty_params = torch.tensor([penvalueridge,
                                           penfirstridge,
                                           pensecondridge,
                                           ctm_pensecondridge])

            lambda_penalty_param_matrix = if_float_create_lambda_penalisation_matrix(lambda_penalty_param, y_train)

            if cross_validation_folds == False:


                y_train = y_train.to(device)
                y_validate = y_validate.to(device)

                if linear_mctm:
                    nf_mctm = MCTM(input_min=y_train.min(0).values,  # y_sub_train
                                   input_max=y_train.max(0).values,  # y_sub_train
                                   polynomial_range=poly_range,
                                   number_variables=y_train.size()[1],  # y_sub_train
                                   spline_transformation=spline_transformation,
                                   spline_decorrelation=spline_decorrelation,
                                   degree_transformations=int(degree_transformations),
                                   degree_decorrelation=int(degree_decorrelation),
                                   span_factor=span_factor,  # device=device torch.tensor(
                                   span_restriction=span_restriction,
                                   number_covariates=number_covariates,
                                   list_comprehension=list_comprehension,
                                   num_decorr_layers=num_decorr_layers,
                                   num_trans_layers=num_trans_layers,
                                   initial_log_transform=initial_log_transform)  # device=device
                elif factor_graph:
                    nf_mctm = Factor_Graph( input_min = y_train.min(0).values, 
                                            input_max = y_train.max(0).values, 
                                            polynomial_range = list([[-5], [5]]),
                                            degree_transformations=int(degree_transformations),
                                            degree_decorrelation=int(degree_decorrelation),
                                            affine_decorr_layer=affine_decorr_layer, 
                                            degree_multi=int(degree_multi))
                else:
                    nf_mctm = NF_MCTM(input_min=y_train.min(0).values,  # y_sub_train
                                      input_max=y_train.max(0).values,  # y_sub_train
                                      polynomial_range=poly_range,
                                      number_variables=y_train.size()[1],  # y_sub_train
                                      spline_transformation=spline_transformation,
                                      spline_decorrelation=spline_decorrelation,
                                      degree_transformations=int(degree_transformations),
                                      degree_decorrelation=int(degree_decorrelation),
                                      span_factor=span_factor,  # device=device torch.tensor(
                                      span_restriction=span_restriction,
                                      number_covariates=number_covariates,
                                      list_comprehension=list_comprehension,
                                      num_decorr_layers=num_decorr_layers,
                                      num_trans_layers=num_trans_layers,
                                      initial_log_transform=initial_log_transform,
                                      calc_method_bspline=calc_method_bspline,
                                      affine_decorr_layer=affine_decorr_layer,
                                      degree_multi=degree_multi)  # device=device

                #nf_mctm = NF_MCTM(input_min=y_train.min(0).values,
                #                  input_max=y_train.max(0).values,
                #                  polynomial_range=poly_range,
                #                  number_variables=y_train.size(1),
                #                  spline_transformation=spline_transformation,
                #                  spline_decorrelation=spline_decorrelation,
                #                  degree_transformations=degree_transformations,
                #                  degree_decorrelation=degree_decorrelation,
                #                  number_covariates=number_covariates,
                #                  span_factor=span_factor,
                #                  span_restriction=span_restriction,
                #                  initial_log_transform=initial_log_transform,
                #                  covaraite_effect=covaraite_effect)
                # normalisation_layer=normalisation_layer)

                if pretrained_transformation_layer == True:
                    
                    nf_mctm.pretrain_tranformation_layer(train_data=y_train,
                                                        validate_data=y_validate_cv,
                                                        train_covariates=x_train_cv,
                                                        validate_covariates=x_validate_cv,
                                                        penalty_params=penalty_params,
                                                        lambda_penalty_params=lambda_penalty_params,
                                                        iterations=iterations, 
                                                        learning_rate=learning_rate,
                                                        patience=patience,
                                                        min_delta=min_delta,
                                                        verbose=False,
                                                        optimizer=optimizer,
                                                        lambda_penalty_mode=lambda_penalty_mode,
                                                        objective_type=objective_type,
                                                        return_plot=False,
                                                        ema_decay=ema_decay)
                        
                    #transformation_model = pretrain_transformation_layer(y_train=y_train,
                    #                                                     poly_span_abs=poly_span_abs,
                    #                                                     degree_transformations=degree_transformations,
                    #                                                     span_factor=span_factor,
                    #                                                     monotonically_increasing=True,
                    #                                                     spline=spline_transformation,
                    #                                                     iterations=iterations,
                    #                                                     patience=patience,
                    #                                                     min_delta=min_delta,
                    #                                                     return_report=False,
                    #                                                     verbose=False,
                    #                                                     initial_log_transform=initial_log_transform,
                    #                                                     objective_type=objective_type,
                    #                                                     pen_ctm_second_ridge=ctm_pensecondridge,
                    #                                                     calc_method_bspline=calc_method_bspline)
                    #nf_mctm.state_dict()["transformation.params"].data.copy_(
                    #    transformation_model.state_dict()["l1.params"])
                    #
                    if match_after_preprocessing == True:
                        nf_mctm.transformation.params.requires_grad = False

                nf_mctm = nf_mctm.to(device)
                
                nf_mctm.train(train_data=y_train,
                            validate_data=y_validate, 
                            train_covariates=x_train,
                            validate_covariates=x_validate,
                            penalty_params=penalty_params.to(device),
                            lambda_penalty_params=lambda_penalty_param_matrix,
                            iterations=iterations, 
                            learning_rate=learning_rate,
                            patience=patience,
                            min_delta=min_delta,
                            verbose=False,
                            optimizer=optimizer,
                            lambda_penalty_mode=lambda_penalty_mode,
                            objective_type=objective_type,
                            return_plot=False,
                            ema_decay=ema_decay)

                #train(model=nf_mctm,
                #      train_data=y_train,
                #      validate_data=y_validate,
                #      train_covariates=x_train,
                #      validate_covariates=x_validate,
                #      penalty_params=penalty_params.to(device),
                #      lambda_penalty_params=lambda_penalty_param_matrix,
                #      iterations=iterations,
                #      learning_rate=learning_rate,
                #      patience=patience,
                #      min_delta=min_delta,
                #      verbose=False,
                #      return_report=False,
                #      objective_type=objective_type,
                #      lambda_penalty_mode=lambda_penalty_mode,
                #      optimizer=optimizer,
                #      ema_decay=ema_decay)  # no need for reporting and metrics,plots etc.
                

                mean_validation_log_likelihood = nf_mctm.log_likelihood(y_validate, x_validate).cpu().detach().numpy().mean()

            else:

                kf = KFold(n_splits=cross_validation_folds, shuffle=True, random_state=random_state_KFold)
                log_likelihoods = []
                for train_idx, val_idx in kf.split(np.arange(y_train.size()[0])):
                    y_train_cv = y_train[train_idx, :]
                    y_validate_cv = y_train[val_idx, :]

                    if x_train is False:
                        x_train_cv = False
                        x_validate_cv = False
                    else:
                        x_train_cv = x_train[train_idx]#, :
                        x_validate_cv = x_train[val_idx]#, :

                    y_train_cv = y_train_cv.to(device)
                    y_validate_cv = y_validate_cv.to(device)

                    if linear_mctm:
                        nf_mctm = MCTM(input_min=y_train.min(0).values,  # y_sub_train
                                       input_max=y_train.max(0).values,  # y_sub_train
                                       polynomial_range=poly_range,
                                       number_variables=y_train.size()[1],  # y_sub_train
                                       spline_transformation=spline_transformation,
                                       spline_decorrelation=spline_decorrelation,
                                       degree_transformations=int(degree_transformations),
                                       degree_decorrelation=int(degree_decorrelation),
                                       span_factor=span_factor,  # device=device torch.tensor(
                                       span_restriction=span_restriction,
                                       number_covariates=number_covariates,
                                       list_comprehension=list_comprehension,
                                       num_decorr_layers=num_decorr_layers,
                                       num_trans_layers=num_trans_layers,
                                       initial_log_transform=initial_log_transform)  # device=device
                    elif factor_graph:
                        nf_mctm = Factor_Graph( input_min = y_train.min(0).values, 
                                                input_max = y_train.max(0).values, 
                                                polynomial_range = list([[-5], [5]]),
                                                degree_transformations=int(degree_transformations),
                                                degree_decorrelation=int(degree_decorrelation),
                                                affine_decorr_layer=affine_decorr_layer, 
                                                degree_multi=int(degree_multi))
                    else:
                        nf_mctm = NF_MCTM(input_min=y_train.min(0).values,  # y_sub_train
                                          input_max=y_train.max(0).values,  # y_sub_train
                                          polynomial_range=poly_range,
                                          number_variables=y_train.size()[1],  # y_sub_train
                                          spline_transformation=spline_transformation,
                                          spline_decorrelation=spline_decorrelation,
                                          degree_transformations=int(degree_transformations),
                                          degree_decorrelation=int(degree_decorrelation),
                                          span_factor=span_factor,  # device=device torch.tensor(
                                          span_restriction=span_restriction,
                                          number_covariates=number_covariates,
                                          list_comprehension=list_comprehension,
                                          num_decorr_layers=num_decorr_layers,
                                          num_trans_layers=num_trans_layers,
                                          initial_log_transform=initial_log_transform,
                                          calc_method_bspline=calc_method_bspline,
                                          affine_decorr_layer=affine_decorr_layer,
                                          degree_multi=degree_multi)  # device=device

                    #nf_mctm = NF_MCTM(input_min=y_train_cv.min(0).values,
                    #                  input_max=y_train_cv.max(0).values,
                    #                  polynomial_range=poly_range,
                    #                  number_variables=y_train_cv.size()[1],
                    #                  spline_decorrelation=spline_decorrelation,
                    #                  degree_transformations=degree_transformations,
                    #                  degree_decorrelation=degree_decorrelation,
                    #                  number_covariates=number_covariates,
                    #                  span_factor=span_factor,
                    #                  span_restriction=span_restriction,
                    #                  initial_log_transform=initial_log_transform,
                    #                  covaraite_effect=covaraite_effect)
                    # normalisation_layer=normalisation_layer)

                    if pretrained_transformation_layer == True:
                        
                        nf_mctm.pretrain_tranformation_layer(train_data=y_train,
                                                            validate_data=y_validate_cv,
                                                            train_covariates=x_train_cv,
                                                            validate_covariates=x_validate_cv,
                                                            penalty_params=penalty_params,
                                                            lambda_penalty_params=lambda_penalty_params,
                                                            iterations=iterations, 
                                                            learning_rate=learning_rate,
                                                            patience=patience,
                                                            min_delta=min_delta,
                                                            verbose=False,
                                                            optimizer=optimizer,
                                                            lambda_penalty_mode=lambda_penalty_mode,
                                                            objective_type=objective_type,
                                                            return_plot=False,
                                                            ema_decay=ema_decay)
                         
                        #transformation_model = pretrain_transformation_layer(y_train=y_train_cv,
                        #                                                     poly_span_abs=poly_span_abs,
                        #                                                     degree_transformations=degree_transformations,
                        #                                                     span_factor=span_factor,
                        #                                                     monotonically_increasing=True,
                        #                                                     spline=spline_transformation,
                        #                                                     iterations=iterations,
                        #                                                     patience=patience,
                        #                                                     min_delta=min_delta,
                        #                                                     return_report=False,
                        #                                                     verbose=False,
                        #                                                     initial_log_transform=initial_log_transform,
                        #                                                     objective_type=objective_type)
                        #nf_mctm.state_dict()["transformation.params"].data.copy_(transformation_model.state_dict()["l1.params"])
                        
                        if match_after_preprocessing == True:
                            nf_mctm.transformation.params.requires_grad = False

                    nf_mctm = nf_mctm.to(device)

                    # print("nf_mctm",nf_mctm.device)
                    # print("y_train_cv", y_train_cv.device)
                    # print("y_validate_cv", y_validate_cv.device)
                    
                    nf_mctm.train(train_data=y_train_cv,
                            validate_data=y_validate_cv, 
                            train_covariates=x_train_cv,
                            validate_covariates=x_validate_cv,
                            penalty_params=penalty_params.to(device),
                            lambda_penalty_params=lambda_penalty_param_matrix,
                            iterations=iterations, 
                            learning_rate=learning_rate,
                            patience=patience,
                            min_delta=min_delta,
                            verbose=False,
                            optimizer=optimizer,
                            lambda_penalty_mode=lambda_penalty_mode,
                            objective_type=objective_type,
                            return_plot=False,
                            ema_decay=ema_decay)

                    #train(model=nf_mctm,
                    #      train_data=y_train_cv,
                    #      validate_data=y_validate_cv,
                    #      train_covariates=x_train_cv,
                    #      validate_covariates=x_validate_cv,
                    #      penalty_params=penalty_params.to(device),
                    #      lambda_penalty_params=lambda_penalty_param_matrix,
                    #      iterations=iterations,
                    #      learning_rate=learning_rate,
                    #      patience=patience,
                    #      min_delta=min_delta,
                    #      verbose=False,
                    #      return_report=False,
                    #      objective_type=objective_type,
                    #      lambda_penalty_mode=lambda_penalty_mode,
                    #      optimizer=optimizer,
                    #      ema_decay=ema_decay)  # no need for reporting and metrics,plots etc.

                    log_likelihoods.append(nf_mctm.log_likelihood(y_validate_cv, x_validate_cv).cpu().detach().numpy().mean())

                mean_validation_log_likelihood = np.mean(log_likelihoods)

            results = results.append({'penvalueridge': penvalueridge, 'penfirstridge': penfirstridge,
                                      'pensecondridge': pensecondridge,
                                      'ctm_pensecondridge': ctm_pensecondridge,
                                      'lambda_penalty_param': lambda_penalty_param,
                                      'learning_rate': learning_rate,
                                      'patience': patience, 'min_delta': min_delta,
                                      'degree_transformations': degree_transformations,
                                      'degree_decorrelation': degree_decorrelation,
                                      #'normalisation_layer': normalisation_layer,
                                      #'fold': fold,
                                      'mean_validation_log_likelihood': mean_validation_log_likelihood},
                                     ignore_index=True)

        print("hyperparameter_tuning done")
        return results


def extract_optimal_hyperparameters(results: pd.DataFrame):
    """
    Extracts the optimal hyperparameters from the results dataframe
    :param results:
    :return:
    """

    results_mean = results.groupby(['penvalueridge', 'penfirstridge', 'pensecondridge', 'ctm_pensecondridge', 'lambda_penalty_param',
                                    'learning_rate',
                                    'patience', 'min_delta', 'degree_transformations',
                                    'degree_decorrelation']).mean()
    results_std = results.groupby(['penvalueridge', 'penfirstridge', 'pensecondridge', 'ctm_pensecondridge', 'lambda_penalty_param',
                                   'learning_rate',
                                    'patience', 'min_delta', 'degree_transformations',
                                    'degree_decorrelation']).std()
    # ODO: cannot grooubby if we have nan values in a column, e.g. dropped normalisation_layer, if I decide to keep this layer then fix this
    results_mean = results_mean.reset_index()
    results_std = results_std.reset_index()

    optimal_hyperparameters = results_mean.loc[results_mean['mean_validation_log_likelihood'].idxmax()]
    if 'fold' in results_mean.columns:
        optimal_hyperparameters = optimal_hyperparameters.drop(['fold'])
    optimal_hyperparameters = optimal_hyperparameters.drop(['mean_validation_log_likelihood'])

    results_mean["std_validation_log_likelihood"] = results_std["mean_validation_log_likelihood"]
    results_mean.rename(columns={'mean_validation_log_likelihood': 'mean_validation_log_likelihood)'}, inplace=True)
    if 'fold' in results_mean.columns:
        results_moments = results_mean.drop(['fold'],axis=1)
    else:
        results_moments = results_mean
    return optimal_hyperparameters.values, results_moments

