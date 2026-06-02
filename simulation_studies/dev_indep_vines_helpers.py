import torch
import numpy as np
import pandas as pd
from scipy import stats
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

def load_sim_infos(seed,
                   train_obs = 4000,
                   copula = "cvine",
                   copula_par = 10):
    #seed=4 #4 #1

    data_folder = "simulation_study_data/" + str(copula) + "_" + str(copula_par) + "_" + str(train_obs) + "/"
    
    family = torch.tensor(pd.read_csv(data_folder + str(seed) + "_family.csv",header=None).values, dtype=torch.float32)
    par = torch.tensor(pd.read_csv(data_folder + str(seed) + "_par.csv",header=None).values, dtype=torch.float32)
    rvinematrix = torch.tensor(pd.read_csv(data_folder + str(seed) + "_rvinematrix.csv",header=None).values, dtype=torch.float32)
    
    #copula = "cvine" #TODO: correct this here, wron kendalltau matrix beacause not stored wanted to just run it
    try:
        data_folder = "simulation_study_data/" + str(copula) + "_" + str(copula_par) + "_" + str(train_obs) + "/"
        kendalltau = torch.tensor(pd.read_csv(data_folder + str(seed) + "_kendalltaumatrix.csv",header=None).values, dtype=torch.float32)
    except:
        warnings.warn("Kendalltau matrix not found, using par matrix instead")
        kendalltau = par
    
    return family, par, rvinematrix, kendalltau


def get_true_ci_graph(family, par, rvinematrix, kendalltau):

    # Initialize lists to store the data
    var_row = []
    var_col = []
    family_list = []
    par_list = []
    kendalltau_list = []
    dependence_list = []
    #rvinematrix_list = []
    D = family.shape[0]
    
    # independence row is indetified hence last row where all copulas are independence copula, needed below for CI identification
    # -1 because row count starts at zero
    independence_row = (family.sum(1) == 0).sum() -1

    # Extract the lower triangular part (excluding the diagonal)
    for i in range(D):
        for j in range(i):
            
            var_row.append(int(rvinematrix[j, j].item() -1)) #diagonal so upper elements give one of the variables
            
            if i > j:
                var_col.append(int(rvinematrix[i, j].item() -1)) #the other variable is the one in the entry 
            elif i == j:
                var_col.append(int(rvinematrix[D, D].item()- 1)) #or if we are at the top then we have tht its the bottom right one
            
            family_list.append(family[i, j].item())
            par_list.append(par[i, j].item())
            kendalltau_list.append(kendalltau[i, j].item())
            
            # if row is the one under the indepdenence row then independence copula is independence, otherwise it is not
            if i == (independence_row+1):
                dependence_list.append(1* (family[i, j].item() > 0))
            elif i > (independence_row+1):
                dependence_list.append(1)
            else:
                dependence_list.append(0)
                
            #rvinematrix_list.append(rvinematrix[i, j].item())

    # Create the DataFrame
    true_ci_graph = pd.DataFrame({
        'var_row': var_row,
        'var_col': var_col,
        'family': family_list,
        'par': par_list,
        'kendalltau': kendalltau_list
    })

    true_ci_graph["dependence"] = dependence_list #1 * (true_ci_graph["family"] > 0) #1 is true, 0 is false
    #true_ci_graph["independence"] = 1 * (true_ci_graph["family"] == 0) #1 is true, 0 is false
    #true_ci_graph["dependence"][:21] = 0
    #true_ci_graph["dependence"][21:] = 1
    # Print the DataFrame
    #print(true_ci_graph)
    
    # Correction for unordered rvines
    true_ci_graph_2 = true_ci_graph.copy()
    true_ci_graph_2["var_row"] = true_ci_graph[["var_row","var_col"]].max(1)
    true_ci_graph_2["var_col"] = true_ci_graph[["var_row","var_col"]].min(1)
    true_ci_graph = true_ci_graph_2
    
    return true_ci_graph

def get_metrics(true_ci_graph, matrix_stats, print_info=True):
    merged_df = true_ci_graph.merge(matrix_stats, on=['var_row', 'var_col'])
    
    pearson_corr_dependence = np.corrcoef(merged_df["kld"],merged_df["dependence"],"pearson")[0,1]
    spearman_corr_dependence = stats.spearmanr(merged_df["kld"],merged_df["dependence"]).statistic
    point_biserial_corr_dependence, p_value = stats.pointbiserialr(merged_df["kld"], merged_df["dependence"])
    
    # Train logistic regression model
    model = LogisticRegression()
    X = merged_df["kld"].to_numpy().reshape(-1, 1)
    y_true = merged_df["dependence"].to_numpy() 
    model.fit(X, y_true)
    y_pred = model.predict(X)
    #y_pred_ind = abs(y_pred - 1) #makes 1 to 0 and 0 to 1
    # Evaluate the model
    #accuracy_dependence = accuracy_score(merged_df["dependence"].to_numpy(), y_pred)
    
    #print(X[30:,:])
    cm = confusion_matrix(y_true, y_pred)
    #print(cm)
    tn, fp, fn, tp = cm.ravel()
    if print_info == True:
        print("tn",tn)
        print("fp",fp)
        print("fn",fn)
        print("tp",tp)
    #print(y_true)
    #print(y_pred)
    accuracy_dependence = tp / (tp + fn)
    accuracy_independence = tn / (tn + fp)
    
    
    pearson_corr_kendalltau = np.corrcoef(merged_df["kld"],merged_df["kendalltau"],"pearson")[0,1]
    spearman_corr_kendalltau = stats.spearmanr(merged_df["kld"],merged_df["kendalltau"]).statistic
    
    #accuracy_independence = accuracy_score(merged_df["independence"].to_numpy(), y_pred_ind)
    
    return pearson_corr_dependence, spearman_corr_dependence, point_biserial_corr_dependence, accuracy_dependence, pearson_corr_kendalltau, spearman_corr_kendalltau, accuracy_independence


import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer

def vuong_test(m, N, mod1_df=2, mod2_df=2, correction=True, print_info=True):
    '''
    mod1, mod2 - non-nested logitstic regression fit results from statsmodels
    '''
    ## number of observations and check of models
    #N = mod1.nobs
    #N2 = mod2.nobs
    #if N != N2:
    #    raise ValueError('Models do not have the same number of observations')
    ## extract the log-likelihood for individual points with the models
    #m1 = mod1.model.loglikeobs(mod1.params)
    #m2 = mod2.model.loglikeobs(mod2.params)
    ## point-wise log likelihood ratio
    #m = m1 - m2
    # calculate the LR statistic
    LR = np.sum(m)
    # calculate the AIC and BIC correction factors -> these go to zero when df is same between models
    AICcor = mod1_df - mod2_df #mod1.df_model - mod2.df_model
    BICcor = np.log(N)*AICcor/2
    # calculate the omega^2 term
    omega2 = np.var(m, ddof=1)
    # calculate the Z statistic with and without corrections
    Zs = np.array([LR,LR-AICcor,LR-BICcor])
    Zs /= np.sqrt(N*omega2)
    # calculate the p-value
    ps = []
    msgs = []
    for Z in Zs:
        if Z>0:
            ps.append(1 - norm.cdf(Z))
            msgs.append('model 1 preferred over model 2')
        else:
            ps.append(norm.cdf(Z))
            msgs.append('model 2 preferred over model 1')
    if print_info == True:
        # share information
        print('=== Vuong Test Results ===')
        labs = ['Uncorrected']
        if AICcor!=0:
            labs += ['AIC Corrected','BIC Corrected']
        for lab,msg,p,Z in zip(labs,msgs,ps,Zs):
            print('  -> '+lab)
            print('    -> '+msg)
            print('    -> Z: '+str(Z))
            print('    -> p: '+str(p))
    

## load sample data
#X,y = load_breast_cancer( return_X_y=True, as_frame=True)
## create data for modeling
#X1 = sm.add_constant( X.loc[:,('mean radius','perimeter error','worst symmetry')])
#X2 = sm.add_constant( X.loc[:,('mean area','worst smoothness')])
## fit the models
#mod1 = sm.Logit( y, X1).fit()
#mod2 = sm.Logit( y, X2).fit()
#
## run Vuong's Test
#vuong_test( mod1, mod2)
#
## save data for R test function
#pd.concat( [X,y], axis=1).to_csv('breast_cancer_data.csv')

import pickle
import matplotlib.pyplot as plt

import scipy
# we need to plot the kld to see what the std is and if the estimates are stable, are the differences distributed without crazy outliers

from sklearn.covariance import GraphicalLassoCV

def run_graph_lasso(seed_value=1,
                    copula="rvine",
                    copula_par=10,
                    train_obs=1000,
                    covariate_exists = False):

    data_folder = "simulation_study_data/" + str(copula) + "_" + str(copula_par) + "_" + str(train_obs) + "/"

    # Getting the training data
    y_train = torch.tensor(pd.read_csv(data_folder + str(seed_value) + "_sample_train.csv").values, dtype=torch.float32)
    y_validate = torch.tensor(pd.read_csv(data_folder + str(seed_value) + "_sample_validate.csv").values, dtype=torch.float32)
    train_log_likelihood = torch.tensor(pd.read_csv(data_folder + str(seed_value) + "_train_log_likelihoods.csv").values, dtype=torch.float32).flatten()

    if "vine" in copula:
        y_test = torch.tensor(pd.read_csv(data_folder + str(seed_value) + "_grid_test.csv").values, dtype=torch.float32)
        test_log_likelihood = torch.tensor(pd.read_csv(data_folder + str(seed_value) + "_test_log_likelihoods.csv").values, dtype=torch.float32).flatten()
    else:
        y_test = torch.tensor(pd.read_csv(data_folder + "grid_test.csv").values, dtype=torch.float32)
        test_log_likelihood = torch.tensor(pd.read_csv(data_folder + "test_log_likelihoods.csv").values, dtype=torch.float32).flatten()
    if covariate_exists == True:
        x_train = torch.tensor(pd.read_csv(data_folder + str(seed_value) + "_train_covariate.csv").values, dtype=torch.float32)
        x_validate = torch.tensor(pd.read_csv(data_folder + str(seed_value) + "_validate_covariate.csv").values, dtype=torch.float32)
        x_train = x_train.squeeze()
        x_validate = x_validate.squeeze()
        x_test = torch.tensor(pd.read_csv(data_folder + "test_covariate.csv").values, dtype=torch.float32)
        x_test = x_test.squeeze()
        number_covariates = 1
    else:
        x_train = False
        x_validate = False
        x_test = False
        number_covariates = 0
        
    Y = y_train.numpy()
    cov = GraphicalLassoCV(verbose=False).fit(Y)
    
    p_matrix = np.around(cov.get_precision(), decimals=3)
    
    # Get the lower triangular indices, including the diagonal
    lower_triangular_indices = np.tril_indices_from(p_matrix, k=-1)

    # Extract the corresponding entries from the precision matrix
    entries = p_matrix[lower_triangular_indices]

    # Create a list of tuples (row_number, col_number, precision_matrix_entry)
    data = list(zip(lower_triangular_indices[0], lower_triangular_indices[1], entries))

    # Convert the list of tuples into a Pandas DataFrame
    df = pd.DataFrame(data, columns=['var_row', 'var_col', 'ggm_p_matrix'])
    
    return df


def run_analysis(subset, dependence_metric="vuong_pvalue", vuong_pvalue_krit=None, copula="cvine", file_kld_appendage = "", kld_available=True, evaluation_data_type="training_data", 
                 print_info=True, sample_size=10000#, recalculate_pmatrix=False
                 ):

    sub_kld_summary_statistics_training_data_list = []

    for index, row in subset.iterrows():
        if print_info == True:
            print("run_id",row["tags.mlflow.runName"])

        artifact_uri = row["artifact_uri"]
        seed = row["tags.seed"]
        train_obs = row["tags.seed"]
        copula_par=row["tags.copula_par"]
        train_obs=row["tags.train_obs"]
        if kld_available == True:
            uri = artifact_uri + "/sub_kld_summary_statistics_{}{}.csv".format(evaluation_data_type, file_kld_appendage)
            sub_kld_summary_statistics_training_data = pd.read_csv(uri)
        
        if evaluation_data_type=="training_data":
            eval_str = ""
        else:
            eval_str = "_" + evaluation_data_type
            
        uri = artifact_uri + "/conditional_correlation_matrix_summary_statistics"+eval_str+".csv"
        sub_condcorr_summary_statistics_training_data = pd.read_csv(uri)
        uri = artifact_uri + "/precision_matrix_summary_statistics"+eval_str+".csv"
        sub_pmatrix_summary_statistics_training_data = pd.read_csv(uri)
        
        
        #############################################################################################################################################
        #############################################################################################################################################
        #############################################################################################################################################
        #if recalculate_pmatrix == True:
        #    # Recalculate the Precision Matrix and Conditional Correlation Matrix
        #    nf_mctm = mlflow.pytorch.load_model(artifact_uri + "/nf_mctm_model_total", map_location=torch.device("cpu"))
        #    #nf_mctm = mlflow.pytorch.load_model("runs:/{}/nf_mctm_model_total".format(row["tags.mlflow.runid"]), map_location=torch.device("cpu"))
        #    nf_mctm.device = "cpu"
        #    data_folder = "simulation_study_data/" + str(row["tags.copula"]) + "_" + str(row["tags.copula_par"]) + "_" + str(row["tags.train_obs"]) + "/"
        #    if evaluation_data_type=="samples_from_model":
        #        
        #        nf_mctm.transformation.approximate_inverse(input = torch.tensor(pd.read_csv(data_folder + str(row["tags.seed"]) + "_sample_train.csv").values, dtype=torch.float32)
        #                                                )
        #        
        #        y_train = nf_mctm.sample(sample_size).detach()
        #    elif evaluation_data_type=="training_data":
        #        y_train = torch.tensor(pd.read_csv(data_folder + str(row["tags.seed"]) + "_sample_train.csv").values, dtype=torch.float32)
        #    precision_matrix_train = nf_mctm.compute_precision_matrix(y_train, covariate=False)
        #    def p_to_corr(matrix):
        #        d = matrix.size(0)
        #        diag_sqrt = torch.diag(matrix) ** 0.5
        #        matrix_std_multiplied = np.matmul(torch.reshape(diag_sqrt, (d, 1)), torch.reshape(diag_sqrt, (1, d)))
        #        return -1 * matrix / matrix_std_multiplied
        #    conditional_correlation_matrix_train = torch.stack([p_to_corr(precision_matrix_train[obs_num,:,:]) for obs_num in range(precision_matrix_train.size(0))])
        #    from python_nf_mctm.simulation_study.simulation_study_helpers import compute_precision_matrix_summary_statistics
        #    sub_pmatrix_summary_statistics_training_data = compute_precision_matrix_summary_statistics(precision_matrix_train)
        #    sub_condcorr_summary_statistics_training_data = compute_precision_matrix_summary_statistics(conditional_correlation_matrix_train)
        #############################################################################################################################################
        #############################################################################################################################################
        #############################################################################################################################################
        
        
        family, par, rvinematrix, kendalltau = load_sim_infos(seed,
                        train_obs = train_obs,
                        copula = copula,
                        copula_par = copula_par)
        true_ci_graph = get_true_ci_graph(family, par, rvinematrix, kendalltau)
        
        #ggm_df = run_graph_lasso(seed_value=row["tags.seed"],
        #            copula=row["tags.copula"],
        #            copula_par=row["tags.copula_par"],
        #            train_obs=row["tags.train_obs"],
        #            covariate_exists = row["params.covariate_exists"])
        
        if kld_available == True:
            with open(artifact_uri[7:]+"/actual_log_distribution_glq_list_{}{}.pkl".format(evaluation_data_type,file_kld_appendage), 'rb') as file:
                actual_log_distribution_glq_list = pickle.load(file)
            with open(artifact_uri[7:]+"/under_ci_assumption_log_distribution_glq_list_{}{}.pkl".format(evaluation_data_type,file_kld_appendage), 'rb') as file:
                under_ci_assumption_log_distribution_glq_list = pickle.load(file)
        
        vuong_stat_list = []
        vuong_cdf_value_list = []
        vuong_pvalue_list = []
        iae_list = []
        ll_dev_square_mean_list = []
        simpletest_pvalue_list = []
        kld_correct_list = []
        ll_dev_abs_mean_list = []
        ll_dev_mean_list = []
        for row_num in range(sub_pmatrix_summary_statistics_training_data.shape[0]):
            var_row_num = int(sub_pmatrix_summary_statistics_training_data.iloc[row_num]["var_row"])
            var_col_num = int(sub_pmatrix_summary_statistics_training_data.iloc[row_num]["var_col"])
            
            if kld_available == True:
                actual_log_distribution_glq = actual_log_distribution_glq_list[row_num]
                under_ci_assumption_log_distribution_glq = under_ci_assumption_log_distribution_glq_list[row_num]
                
                ll_dev = actual_log_distribution_glq - under_ci_assumption_log_distribution_glq
                
                #if sum(torch.isnan(ll_dev))/ll_dev.size(0) > 0:
                #    print("Percentage of NaN in ll_dev", sum(torch.isnan(ll_dev))/ll_dev.size(0))
                #if sum(torch.isinf(ll_dev))/ll_dev.size(0) > 0:
                #    print("Percentage of Inf in ll_dev", sum(torch.isinf(ll_dev))/ll_dev.size(0))
                
                bool_inf_nan = torch.isnan(ll_dev) | torch.isinf(ll_dev)
                
                ll_dev = ll_dev[~bool_inf_nan]
                
                l_abs_dev = torch.abs(torch.exp(actual_log_distribution_glq) - torch.exp(under_ci_assumption_log_distribution_glq))
                
                # needed for iae calculation
                # needs to be before the nan and inf removal of l_abs_dev
                actual_log_distribution_glq_sub = actual_log_distribution_glq[~(torch.isinf(l_abs_dev) | torch.isnan(l_abs_dev))]
                
                l_abs_dev = l_abs_dev[~torch.isnan(l_abs_dev)]
                l_abs_dev = l_abs_dev[~torch.isinf(l_abs_dev)]
                
                #print(ll_dev.size(0))
                
                # IAE is the integral over random samples hence by using the training samples we need to divide by the probability of the samples 
                # to get back to random sampling. This is crude because we dont have the true probability of the sampling only the approx from the model
                iae_obs = l_abs_dev / torch.exp(actual_log_distribution_glq_sub) 
                iae_obs = iae_obs[~(torch.isinf(iae_obs) | torch.isnan(iae_obs))]
                iae_list.append(iae_obs.mean())
                
                
                kld = torch.exp(actual_log_distribution_glq[~bool_inf_nan]) * ll_dev
                
                bool_inf_nan = torch.isnan(kld) | torch.isinf(kld)
                kld = kld[~bool_inf_nan]
                
                kld_correct_list.append(kld.mean())
                
                ll_dev_abs_mean_list.append(ll_dev.abs().mean())
                ll_dev_mean_list.append(ll_dev.mean())
            
            #plt.hist(ll_dev)
            #plt.show()
            
            #print(sum(torch.isnan(ll_dev)))
            #print(sum(torch.isinf(ll_dev)))
            #print(ll_dev.mean())
            #print(ll_dev.std())
            
            if kld_available == True:
                #if sum(torch.isnan(ll_dev))/ll_dev.size(0) > 0:
                #    print("Percentage of NaN in ll_dev", sum(torch.isnan(ll_dev))/ll_dev.size(0))
                #if sum(torch.isinf(ll_dev))/ll_dev.size(0) > 0:
                #    print("Percentage of Inf in ll_dev", sum(torch.isinf(ll_dev))/ll_dev.size(0))
                
                #ll_dev = ll_dev[~torch.isnan(ll_dev)]
                #ll_dev = ll_dev[~torch.isinf(ll_dev)]
                
                #vuong_test_stat = ll_dev.mean() / ( (ll_dev - ll_dev.mean()).square().sum().sqrt() )
                #vuong_test_stat = ll_dev.mean() / ( ll_dev.square().mean().sqrt() )
                vuong_test_stat = ll_dev.mean() / ( ll_dev.std() )
                #vuong_test_stat = ll_dev.median()
                vuong_cdf_value = torch.distributions.Normal(0.0, 1.0).cdf(torch.FloatTensor([vuong_test_stat]))
                vuong_pvalue = 1 - vuong_cdf_value
                #if vuong_cdf_value > 0.5:
                #    vuong_pvalue = 1 - vuong_cdf_value
                #else:
                #    vuong_pvalue = vuong_cdf_value 
                
                ll_dev_square_mean = ll_dev.square().mean()
            
            
                #vuong_test_stat = ( ll_dev / ( ll_dev.abs().mean() ) ).square().mean()
                #vuong_test_stat = ( ll_dev / ( ll_dev.std() ) ).square().mean()
                #vuong_cdf_value = torch.distributions.chi2.Chi2(1.0).cdf(torch.FloatTensor([vuong_test_stat])) #ll_dev.size(0) before df=1.0
                #vuong_pvalue = 1- vuong_cdf_value
                
                #vuong_test_stat = ll_dev.square().mean() #.var() # #ll_dev.size(0) * ll_dev.var() # before df=1.0
                #vuong_cdf_value = torch.distributions.chi2.Chi2(1.0).cdf(torch.FloatTensor([vuong_test_stat])) #ll_dev.size(0) before df=1.0
                #vuong_pvalue = 1- vuong_cdf_value
                
            if kld_available == True:
                len_obs = actual_log_distribution_glq.size(0)
                
                if len_obs % 2 == 0:
                    pass
                else:
                    len_obs = len_obs-1
                    actual_log_distribution_glq = actual_log_distribution_glq[:-1]
                    under_ci_assumption_log_distribution_glq = under_ci_assumption_log_distribution_glq[:-1]
                
                half_len_obs = int(len_obs/2)
                list_obs = list(range(len_obs))    
                list_obs_even = list_obs[::2]
                list_obs_odd = list_obs[1::2]                
                                                                
                f_a = actual_log_distribution_glq[list_obs_even]
                f_b = under_ci_assumption_log_distribution_glq[list_obs_odd]
                
                ll_deviance = f_a - f_b
                
                ll_deviance = ll_deviance[~torch.isnan(ll_deviance)]
                ll_deviance = ll_deviance[~torch.isinf(ll_deviance)]
                
                var = ll_deviance.square().mean()
                #var = ll_deviance.var()
                
                t = torch.sqrt(torch.FloatTensor([half_len_obs])) * torch.mean(ll_deviance) / torch.sqrt(var)
                
                cdf = torch.distributions.Normal(0,1).cdf(t)
                
                simpletest_pvalue = 1-cdf.item() #min(cdf.item(), 1-cdf.item())
 
            #print(vuong_cdf_value)
            if kld_available == True:
                vuong_stat_list.append(np.round(vuong_test_stat,4).item())
                vuong_cdf_value_list.append(np.round(vuong_cdf_value,4).item())  
                vuong_pvalue_list.append(np.round(vuong_pvalue,4).item())
                ll_dev_square_mean_list.append(np.round(ll_dev_square_mean,4).item())
                simpletest_pvalue_list.append(np.round(simpletest_pvalue,4).item())
            
            #ll_pos = torch.sum(1 *(ll_dev > 0))
            #vuong_cdf_value = scipy.stats.binom.cdf(k=ll_pos.numpy(), n=ll_dev.size(0), p=0.5)
            #if vuong_cdf_value > 0.5:
            #    vuong_pvalue = 1 - vuong_cdf_value
            #else:
            #    vuong_pvalue = vuong_cdf_value 
            #vuong_cdf_value_list.append(vuong_cdf_value.item())  
            #vuong_pvalue_list.append(vuong_pvalue.item())
            
            ##print(vuong_test(m=ll_dev.numpy(), N=ll_dev.size(0), mod1_df=2, mod2_df=2))
            #LR = np.sum(ll_dev.numpy())
            ### calculate the omega^2 term
            #omega2 = np.var(ll_dev.numpy())#, ddof=1)
            ### calculate the Z statistic with and without corrections
            #N = ll_dev.size(0)
            #Z = LR/ np.sqrt(N*omega2)
            ### calculate the p-value
            #if Z>0:
            #    ps = (1 - norm.cdf(Z))
            #    #print('model 1 preferred over model 2, hence dependence')
            #else:
            #    ps = (norm.cdf(Z))
            #    
            #vuong_cdf_value_list.append(norm.cdf(Z).round(3))  
            #vuong_pvalue_list.append(np.round(ps,4))
            

        #plt.hist(vuong_cdf_value_list)
        #plt.show()
        
        sub_pmatrix_summary_statistics_training_data['p_matrix_square_mean'] = sub_pmatrix_summary_statistics_training_data['std']**2 + sub_pmatrix_summary_statistics_training_data['mean']**2
        sub_pmatrix_summary_statistics_training_data = sub_pmatrix_summary_statistics_training_data[["var_row", "var_col", "abs_mean", "p_matrix_square_mean"]]
        
        
        if kld_available == True:
            sub_pmatrix_summary_statistics_training_data["iae"] = iae_list

            sub_pmatrix_summary_statistics_training_data["vuong_stat"] = vuong_stat_list
            sub_pmatrix_summary_statistics_training_data["vuong_pvalue"] = vuong_pvalue_list
            sub_pmatrix_summary_statistics_training_data["vuong_cdf_value"] = vuong_cdf_value_list
            sub_pmatrix_summary_statistics_training_data["ll_dev_square_mean"] = ll_dev_square_mean_list
            sub_pmatrix_summary_statistics_training_data["simpletest_pvalue"] = simpletest_pvalue_list
        
        sub_pmatrix_summary_statistics_training_data = pd.merge(sub_pmatrix_summary_statistics_training_data, true_ci_graph, on=['var_row', 'var_col'])
            
        sub_condcorr_summary_statistics_training_data['condcorr_abs_mean'] = sub_condcorr_summary_statistics_training_data['abs_mean']
        sub_condcorr_summary_statistics_training_data['condcorr_square_mean'] = sub_condcorr_summary_statistics_training_data['std']**2 + sub_condcorr_summary_statistics_training_data['mean']**2
        sub_condcorr_summary_statistics_training_data = sub_condcorr_summary_statistics_training_data[['var_row', 'var_col', 'condcorr_abs_mean', 'condcorr_square_mean']]
        sub_pmatrix_summary_statistics_training_data = pd.merge(sub_pmatrix_summary_statistics_training_data, sub_condcorr_summary_statistics_training_data, on=['var_row', 'var_col'])
        
        
        sub_pmatrix_summary_statistics_training_data['pmatrix_abs_mean'] = sub_pmatrix_summary_statistics_training_data['abs_mean']
        sub_pmatrix_summary_statistics_training_data.drop(columns=['abs_mean'], inplace=True)
        
        if kld_available == True:
            #sub_pmatrix_summary_statistics_training_data = sub_pmatrix_summary_statistics_training_data[['var_row', 'var_col', 'pmatrix_abs_mean']]
            sub_pmatrix_summary_statistics_training_data = pd.merge(sub_pmatrix_summary_statistics_training_data, sub_kld_summary_statistics_training_data[["var_row","var_col","kld"]], on=['var_row', 'var_col'])
            
            sub_pmatrix_summary_statistics_training_data["kld_correct"] = kld_correct_list
            sub_pmatrix_summary_statistics_training_data["ll_dev_abs_mean"] = ll_dev_abs_mean_list
            sub_pmatrix_summary_statistics_training_data["ll_dev_mean"] = ll_dev_mean_list
    
            
        #print(sub_kld_summary_statistics_training_data.sort_values(by=['vuong_cdf_value'],ascending=False)[['var_row','var_col',"family", 'vuong_pvalue_var','vuong_cdf_value']]) #kld. dependence
        
        #print(stats.pointbiserialr(sub_kld_summary_statistics_training_data["vuong_cdf_value"], sub_kld_summary_statistics_training_data["dependence"]))
        
        
        #if seed == "2":
        #    # independence copula in 1,2
        #    sub_kld_summary_statistics_training_data.loc[
        #        (sub_kld_summary_statistics_training_data['var_row'] == 1) & 
        #        (sub_kld_summary_statistics_training_data['var_col'] == 0), 
        #        'dependence'] = 1
        #    # independence copula in 2,6;1
        #    sub_kld_summary_statistics_training_data.loc[
        #        (sub_kld_summary_statistics_training_data['var_row'] == 5) & 
        #        (sub_kld_summary_statistics_training_data['var_col'] == 1), 
        #        'dependence'] = 1
        #if seed == "3":
        #    # independence copula in 2,6;1 is actuall idependent because in the connection in vine above there is an independence
        #    #sub_kld_summary_statistics_training_data.loc[
        #    #    (sub_kld_summary_statistics_training_data['var_row'] == 5) & 
        #    #    (sub_kld_summary_statistics_training_data['var_col'] == 1), 
        #    #    'dependence'] = 1
        #    pass
        #if seed == "4":
        #    # independence copula in 2,7;1 
        #    sub_kld_summary_statistics_training_data.loc[
        #        (sub_kld_summary_statistics_training_data['var_row'] == 6) & 
        #        (sub_kld_summary_statistics_training_data['var_col'] == 1), 
        #        'dependence'] = 1
        
        #if kld_available == True:
        #    # Train logistic regression model
        #    model = LogisticRegression()
        #    #X = sub_kld_summary_statistics_training_data["vuong_pvalue_var"].to_numpy().reshape(-1, 1)
        #    X = sub_pmatrix_summary_statistics_training_data[dependence_metric].to_numpy().reshape(-1, 1)
        #    y_true = sub_pmatrix_summary_statistics_training_data["dependence"].to_numpy() 
        #    model.fit(X, y_true)
        #    y_pred = model.predict(X)
        #    y_pred_ind = abs(y_pred - 1) #makes 1 to 0 and 0 to 1
        #    # Evaluate the model
        #    #accuracy_dependence = accuracy_score(merged_df["dependence"].to_numpy(), y_pred)
        #    
        #    #print(X[30:,:])
        #    if vuong_pvalue_krit is None:
        #        pass
        #    else:
        #        y_pred = sub_pmatrix_summary_statistics_training_data["vuong_pvalue"] < vuong_pvalue_krit #0.28 vuong_cdf_value
        #        
        #    sub_pmatrix_summary_statistics_training_data["y_pred"] = y_pred
        #    cm = confusion_matrix(y_true, y_pred)
        #    #print(cm)
        #    tn, fp, fn, tp = cm.ravel()
        #    #print("tn",tn)
        #    #print("fp",fp)
        #    #print("fn",fn)
        #    #print("tp",tp)
        #    if print_info == True:
        #        print(pd.DataFrame(cm, columns=["t", "f"], index=["t", "f"]))
        #    #print(y_true)
        #    #print(y_pred)
        #    accuracy_dependence = tp / (tp + fn)
        #    accuracy_independence = tn / (tn + fp)
        #    #print("accuracy_dependence",accuracy_dependence.round(2))
        #    #print("accuracy_independence",accuracy_independence.round(2))
            

        
        #print(sub_kld_summary_statistics_training_data.sort_values(by=['vuong_pvalue_var'],ascending=True)[['var_row','var_col',"family", 'vuong_pvalue_var']]) #kld. dependence
        
        #pearson_corr_dependence, spearman_corr_dependence, point_biserial_corr_dependence, accuracy_dependence, pearson_corr_kendalltau, spearman_corr_kendalltau, accuracy_independence = get_metrics(true_ci_graph, sub_kld_summary_statistics_training_data)
        #print(pearson_corr_dependence)
        #print(spearman_corr_dependence)
        #print(point_biserial_corr_dependence)
        #print("accuracy_dependence",accuracy_dependence)
        #print(pearson_corr_kendalltau)
        #print(spearman_corr_kendalltau)
        #print("accuracy_independence",accuracy_independence)
        
        # add results of ggm
        #sub_pmatrix_summary_statistics_training_data = pd.merge(sub_pmatrix_summary_statistics_training_data, ggm_df, on=['var_row', 'var_col'])

        sub_kld_summary_statistics_training_data_list.append(sub_pmatrix_summary_statistics_training_data)
        
    return sub_kld_summary_statistics_training_data_list


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

def plot_roc_curve(true_y, y_prob,label=None,color=None,linestyle=None):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    auc = roc_auc_score(true_y, y_prob)
    plt.plot(fpr, tpr, label=label, color=color,linestyle=linestyle)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    return auc