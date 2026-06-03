from dev_indep_vines_helpers import *
from demos.pyvinecopulib_simulation_helpers import *


def plot_auc_score_boxplot_experiment(experiment_name="rvine_10_dim", 
                                      name_ending_look_for=False, 
                                      name_endings_not_look_for_list=False,
                                      seed_min=False, 
                                      seed_max=False,
                                      kld_available=False,
                                      independence_indetifier_metric="condcorr_abs_mean",
                                      print_info=False,
                                      print_run_names=False,
                                      copula=False):
    
    experiment = mlflow.get_experiment_by_name(experiment_name) 
    df_runs = mlflow.search_runs([experiment.experiment_id])
    subset = df_runs
    
    if name_ending_look_for is not False:
        subset = subset[[name_ending_look_for in subset.iloc[i]["tags.mlflow.runName"] for i in range(subset.shape[0])]]
        
    if name_endings_not_look_for_list is not False:
        for name_ending_not_look_for in name_endings_not_look_for_list:
            subset = subset[[name_ending_not_look_for not in subset.iloc[i]["tags.mlflow.runName"] for i in range(subset.shape[0])]]
    
    if seed_min is not False:
        subset = subset[[int(subset.iloc[i]["tags.seed"]) >= seed_min for i in range(subset.shape[0])]]
    if seed_max is not False:
        subset = subset[[int(subset.iloc[i]["tags.seed"]) <= seed_max for i in range(subset.shape[0])]]
    

    subset_pen = subset#[0:15] # if the most recent seed is still running
    if print_run_names is True: 
        print(subset_pen["tags.mlflow.runName"].to_numpy())
        print(subset_pen["tags.mlflow.runName"].to_numpy().shape)
    
    if copula is False:
        copula = experiment_name[0:5]
    sub_kld_summary_statistics_training_data_list_pen = run_analysis(subset_pen[~subset_pen["metrics.kl_divergence_nf_mctm_test"].isna()], 
                                                                 copula=copula, kld_available=kld_available, #True
                                                                 file_kld_appendage = "_numquadp_20", print_info=print_info)
    
    list_num = 0
    list_auc = []
    for i in range(len(sub_kld_summary_statistics_training_data_list_pen)):
        true_y = sub_kld_summary_statistics_training_data_list_pen[i]["dependence"]
        y_prob = sub_kld_summary_statistics_training_data_list_pen[i][independence_indetifier_metric]
        fpr, tpr, thresholds = roc_curve(true_y, y_prob)
        auc = roc_auc_score(true_y, y_prob)
        list_auc.append(auc)
        
    auc_pen = np.array(list_auc)
    
    return auc_pen

import warnings 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
import mlflow
import logging
mlflow_logger = logging.getLogger("mlflow")
mlflow_logger.setLevel(logging.ERROR)

def analyze_experiment(experiment_name_list,
                       name_ending_look_for1="_correct_pmatrix", 
                       name_ending_look_for2="_no_lambda_pen",
                       independence_indetifier_metric="condcorr_abs_mean",
                       kld_available=False,
                       print_run_names=False,
                       copula=False,
                       seed_min=False, 
                       seed_max=False,
                       plot=True):
    
    list_diffs = []
    for experiment_name in experiment_name_list:
    
        auc_pen = plot_auc_score_boxplot_experiment(experiment_name=experiment_name, 
                                            name_ending_look_for=name_ending_look_for1, 
                                            name_endings_not_look_for_list=[name_ending_look_for2],
                                            seed_min=seed_min, 
                                            seed_max=seed_max,
                                            kld_available=kld_available,
                                            independence_indetifier_metric=independence_indetifier_metric,
                                            print_info=False,
                                            print_run_names=print_run_names,
                                            copula=copula)
        
        auc_no_pen = plot_auc_score_boxplot_experiment(experiment_name=experiment_name, 
                                            name_ending_look_for=name_ending_look_for2, 
                                            name_endings_not_look_for_list=False,
                                            seed_min=seed_min, 
                                            seed_max=seed_max,
                                            kld_available=kld_available,
                                            independence_indetifier_metric=independence_indetifier_metric,
                                            print_info=False,
                                            print_run_names=print_run_names,
                                            copula=copula)
        
        print("%\ improvement",independence_indetifier_metric," auc",experiment_name,":",round(((auc_pen - auc_no_pen)/(auc_no_pen-0.5)).mean() * 100,3))
        print("absolute improvement",independence_indetifier_metric," auc",experiment_name,":",round((auc_pen - auc_no_pen).mean(),4))
        
        list_diffs.append(auc_pen - auc_no_pen)
    
    positions = list(range(len(experiment_name_list)))
    widths = [0.75 for _ in range(len(experiment_name_list))]
    fig, ax = plt.subplots()
    boxplot = ax.boxplot(list_diffs, positions=positions, widths=widths)
    
    for median, pos in zip(boxplot['medians'], positions):
        median_value = median.get_ydata()[0]
        ax.text(pos, median_value, f'{median_value:.3f}', horizontalalignment='center', verticalalignment='center')
    if plot == True:
        plt.show()
    
def analyze_experiment_no_lambda_pen_only(experiment_name_list,
                       name_ending_look_for1="_no_lambda_pen",
                       independence_indetifier_metric="condcorr_abs_mean",
                       kld_available=False,
                       print_run_names=False,
                       copula=False,
                       plot=True):
    
    list_aucs = []
    for experiment_name in experiment_name_list:
    
        auc = plot_auc_score_boxplot_experiment(experiment_name=experiment_name, 
                                            name_ending_look_for=name_ending_look_for1, 
                                            name_endings_not_look_for_list=False,
                                            seed_min=False, 
                                            seed_max=False,
                                            kld_available=kld_available,
                                            independence_indetifier_metric=independence_indetifier_metric,
                                            print_info=False,
                                            print_run_names=print_run_names,
                                            copula=copula)
        
        print("average",independence_indetifier_metric," auc",experiment_name,":",round(auc.mean(),3))
        
        list_aucs.append(auc)
    
    if plot == True:
    
        positions = list(range(len(experiment_name_list)))
        widths = [0.75 for _ in range(len(experiment_name_list))]
        fig, ax = plt.subplots()
        boxplot = ax.boxplot(list_aucs, positions=positions, widths=widths)
        
        for median, pos in zip(boxplot['medians'], positions):
            median_value = median.get_ydata()[0]
            ax.text(pos, median_value, f'{median_value:.3f}', horizontalalignment='center', verticalalignment='center')
        
        plt.show()
    

import mlflow
from mlflow.tracking import MlflowClient

def rename_runs_with_substring(experiment_name, old_substring, new_substring):
    # Initialize MLflow client
    client = MlflowClient()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' does not exist.")
        return
    
    experiment_id = experiment.experiment_id

    # Search all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Loop through each run to check and update the name
    for _, run in runs.iterrows():
        run_id = run.run_id
        run_name = run.get("tags.mlflow.runName", "")  # Retrieve run name from tags
        
        # Check if the name contains the old substring
        if old_substring in run_name:
            # Replace old substring with new substring
            new_name = run_name.replace(old_substring, new_substring)
            
            # Set the updated name with `set_tag`
            client.set_tag(run_id, "mlflow.runName", new_name)
            print(f"Renamed run '{run_name}' to '{new_name}'")
        #else:
        #    print(f"No change needed for run '{run_name}'")

# Example usage
#rename_runs_with_substring("tests3", "no_lambda_pen_correct", "")

import mlflow
from mlflow.tracking import MlflowClient

def append_to_run_names_with_exclusions(experiment_name, append_string, exclusions):
    # Initialize MLflow client
    client = MlflowClient()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' does not exist.")
        return
    
    experiment_id = experiment.experiment_id

    # Search all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Loop through each run to check and update the name
    for _, run in runs.iterrows():
        run_id = run.run_id
        run_name = run.get("tags.mlflow.runName", "")  # Retrieve run name from tags

        # Check if any exclusion substring is in the run name
        if any(exclusion in run_name for exclusion in exclusions):
            #print(f"Skipping run '{run_name}' due to exclusion rule.")
            continue

        # Append the specified string to the run name
        new_name = run_name + append_string
        client.set_tag(run_id, "mlflow.runName", new_name)
        print(f"Updated run name to '{new_name}'")
        
# Example usage
#append_to_run_names_with_exclusions(
#    experiment_name="rvine_weak_10_dim_250obs",
#    append_string="_correct_pmatrix",
#    exclusions=["_adaptive_lasso_resampleall", "_no_lambda_pen", "_lasso_sample_lambda_only"]
#)

def get_auc_across_metrics_experiment(experiment_name="rvine_10_dim", 
                                      name_ending_look_for=False, 
                                      name_endings_not_look_for_list=False,
                                      seed_min=False, 
                                      seed_max=False,
                                      kld_available=False,
                                      independence_indetifier_metrics=["condcorr_abs_mean"],
                                      print_info=False,
                                      print_run_names=False,
                                      copula=False,
                                      file_kld_appendage_train = "_numquadp_20",
                                      file_kld_appendage_synth = "_numquadp_10"):
    
    experiment = mlflow.get_experiment_by_name(experiment_name) 
    df_runs = mlflow.search_runs([experiment.experiment_id])
    subset = df_runs
    
    if name_ending_look_for is not False:
        subset = subset[[name_ending_look_for in subset.iloc[i]["tags.mlflow.runName"] for i in range(subset.shape[0])]]
        
    if name_endings_not_look_for_list is not False:
        for name_ending_not_look_for in name_endings_not_look_for_list:
            subset = subset[[name_ending_not_look_for not in subset.iloc[i]["tags.mlflow.runName"] for i in range(subset.shape[0])]]
    
    if seed_min is not False:
        subset = subset[[int(subset.iloc[i]["tags.seed"]) >= seed_min for i in range(subset.shape[0])]]
    if seed_max is not False:
        subset = subset[[int(subset.iloc[i]["tags.seed"]) <= seed_max for i in range(subset.shape[0])]]
    

    subset_pen = subset#[0:15] # if the most recent seed is still running
    if print_run_names is True: 
        print(subset_pen["tags.mlflow.runName"].to_numpy())
        print(subset_pen["tags.mlflow.runName"].to_numpy().shape)
    
    if copula is False:
        copula = experiment_name[0:5]
    sub_kld_summary_statistics_training_data_list_run_analysis_train = run_analysis(subset_pen, #[~subset_pen["metrics.kl_divergence_nf_mctm_test"].isna()], 
                                                                 copula=copula, kld_available=kld_available, #True
                                                                 file_kld_appendage = file_kld_appendage_train, 
                                                                 evaluation_data_type="training_data", 
                                                                 print_info=print_info)
    
    sub_kld_summary_statistics_training_data_list_run_analysis_synth = run_analysis(subset_pen, #[~subset_pen["metrics.kl_divergence_nf_mctm_test"].isna()], 
                                                                 copula=copula, kld_available=kld_available, #True
                                                                 file_kld_appendage = file_kld_appendage_synth, 
                                                                 evaluation_data_type="samples_from_model", 
                                                                 print_info=print_info)
    
    if "simple_test_opt_epsilon" in independence_indetifier_metrics:
            sub_kld_summary_statistics_training_data_list_simple_test = run_simple_test(subset=subset_pen, 
                copula=copula,
                file_kld_appendage = file_kld_appendage_train, 
                evaluation_data_type="training_data", 
                print_info=print_info)
    
    auc_dict = {}
    for independence_indetifier_metric in independence_indetifier_metrics:
        
        if independence_indetifier_metric == "simple_test_opt_epsilon":
            auc_mean_list = []
            auc_list_list = []
            for epsilon in range(0,21,1):
                epsilon = epsilon/20
                
                auc_list = []
                for i in range(len(sub_kld_summary_statistics_training_data_list_simple_test)):
                    true_y = sub_kld_summary_statistics_training_data_list_simple_test[i]["dependence"]
                    y_prob = 1 - sub_kld_summary_statistics_training_data_list_simple_test[i]["simpletest_pvalue_epsilon_"+str(epsilon)]
                    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
                    auc = roc_auc_score(true_y, y_prob)
                    
                    auc_list.append(auc)
                
                auc_mean_list.append(np.mean(auc_list).item())
                auc_list_list.append(auc_list)

            max_auc_mean = max(auc_mean_list)
            max_auc_mean_index = auc_mean_list.index(max_auc_mean)
            auc_opt_list = auc_list_list[max_auc_mean_index]
                        
            auc_dict["simple_test_opt_epsilon"] = auc_opt_list
            
        else:
            
            type_sum_stats=0 # use this to idetify if train or test set, quick fix
            for sum_stats in [sub_kld_summary_statistics_training_data_list_run_analysis_train, sub_kld_summary_statistics_training_data_list_run_analysis_synth]:
                list_auc = []
                for i in range(len(sum_stats)):
                    true_y = sum_stats[i]["dependence"]
                    y_prob = sum_stats[i][independence_indetifier_metric]
                    # set inf values to the max value for y_prob (solves issue of sometimes seeing inf values for iae)
                    y_prob[y_prob == np.inf] = np.max(y_prob[y_prob != np.inf])
                    # set negative inf values to the min value for y_prob (solves issue of sometimes seeing inf values for iae)
                    y_prob[y_prob == -np.inf] = np.min(y_prob[y_prob != -np.inf])
                    #fpr, tpr, thresholds = roc_curve(true_y, y_prob)
                    # try else print("no auc")
                    try:
                        auc = roc_auc_score(true_y, y_prob)
                        list_auc.append(auc)
                    except:
                        print("no auc for ",independence_indetifier_metric, " in ",experiment_name, " for ",subset_pen.iloc[i]["tags.mlflow.runName"]," for (1=synth, 0=train)", type_sum_stats)
                    #auc = roc_auc_score(true_y, y_prob)
                    #list_auc.append(auc)
                
                    if type_sum_stats == 0:
                        auc_dict[independence_indetifier_metric+"_train"] = list_auc
                    elif type_sum_stats == 1:
                        auc_dict[independence_indetifier_metric+"_synth"] = list_auc
                        
                type_sum_stats += 1
        
    
    return auc_dict

def get_rel_kld_experiment(experiment_name="rvine_10_dim", 
                                      name_ending_look_for=False, 
                                      name_endings_not_look_for_list=False,
                                      seed_min=False, 
                                      seed_max=False,
                                      kld_available=False,
                                      print_info=False,
                                      print_run_names=False,
                                      copula=False):
    
    experiment = mlflow.get_experiment_by_name(experiment_name) 
    df_runs = mlflow.search_runs([experiment.experiment_id])
    subset = df_runs
    
    if name_ending_look_for is not False:
        subset = subset[[name_ending_look_for in subset.iloc[i]["tags.mlflow.runName"] for i in range(subset.shape[0])]]
        
    if name_endings_not_look_for_list is not False:
        for name_ending_not_look_for in name_endings_not_look_for_list:
            subset = subset[[name_ending_not_look_for not in subset.iloc[i]["tags.mlflow.runName"] for i in range(subset.shape[0])]]
    
    if seed_min is not False:
        subset = subset[[int(subset.iloc[i]["tags.seed_value"]) >= seed_min for i in range(subset.shape[0])]]
    if seed_max is not False:
        subset = subset[[int(subset.iloc[i]["tags.seed_value"]) <= seed_max for i in range(subset.shape[0])]]
    

    subset_pen = subset#[0:15] # if the most recent seed is still running
    if print_run_names is True: 
        print(subset_pen["tags.mlflow.runName"].to_numpy())
        print(subset_pen["tags.mlflow.runName"].to_numpy().shape)
    
    
    dict_rel_kld = {}

    list_rel_kld_test = []
    list_rel_kld_train = []
    for i in range(len(subset)):
        subset_pen = subset.iloc[i]
        test_rel_kld_pen = (subset_pen["metrics.kl_divergence_nf_mctm_test"] - subset_pen["metrics.kl_divergence_true_model_test"]) / (subset_pen["metrics.kl_divergence_mvn_model_test"] - subset_pen["metrics.kl_divergence_true_model_test"])
        train_rel_kld_pen = (subset_pen["metrics.kl_divergence_nf_mctm_train"] - subset_pen["metrics.kl_divergence_true_model_train"]) / (subset_pen["metrics.kl_divergence_mvn_model_train"] - subset_pen["metrics.kl_divergence_true_model_train"])
        
        list_rel_kld_test.append(test_rel_kld_pen) 
        list_rel_kld_train.append(train_rel_kld_pen)
        
    dict_rel_kld["rel_kld_test"] = list_rel_kld_test
    dict_rel_kld["rel_kld_train"] = list_rel_kld_train
    
    return dict_rel_kld

def format_data(data_dict, metric, reverse=False):
    formatted_data = []
    for experiment_name in data_dict:
        if reverse == True:
            if metric == False:
                data = 1 - np.array(data_dict[experiment_name])
            else:
                data = 1 - np.array(data_dict[experiment_name][metric])
        else:
            if metric == False:
                data = np.array(data_dict[experiment_name])
            else:
                data = np.array(data_dict[experiment_name][metric])
        
        mean = np.mean(data)
        variance = np.std(data)
        formatted_data.append(f"{mean:.2f} ({variance:.2f})")
        
    return formatted_data

def format_data_differencing(data_dict1, data_dict2, metric, multi=1, relative=False, metric2=False):
    formatted_data = []
    for experiment_name in data_dict1:
        if metric2 == True:
            len_min = min(len(data_dict1[experiment_name][metric]), len(data_dict2[experiment_name]))
            diffs = np.array(data_dict1[experiment_name][metric][0:len_min]) - np.array(data_dict2[experiment_name][0:len_min])
            
            if relative == True:
                diffs = diffs / np.array(data_dict2[experiment_name][0:len_min])
            
        elif metric2 == False:
            len_min = min(len(data_dict1[experiment_name][metric]), len(data_dict2[experiment_name][metric]))
            diffs = np.array(data_dict1[experiment_name][metric][0:len_min]) - np.array(data_dict2[experiment_name][metric][0:len_min])
            
            if relative == True:
                diffs = diffs / np.array(data_dict2[experiment_name][metric][0:len_min])
        
        else:
            len_min = min(len(data_dict1[experiment_name][metric]), len(data_dict2[experiment_name][metric2]))
            diffs = np.array(data_dict1[experiment_name][metric][0:len_min]) - np.array(data_dict2[experiment_name][metric2][0:len_min])
            
            if relative == True:
                diffs = diffs / np.array(data_dict2[experiment_name][metric][0:len_min])
            
        diffs = multi * diffs
        mean = np.mean(diffs)
        std = np.std(diffs) 
        median = np.median(diffs)
        formatted_data.append(f"{mean:.2f}/{median:.2f} ({std:.2f},)")
        
    return formatted_data

import pandas as pd
from tabulate import tabulate

# Function to print LaTeX table
def dataframe_to_latex(df):
    latex_table = tabulate(df, headers='keys', tablefmt='latex')#, showindex=False)
    print(latex_table)
    
    
#define the test value getting function given th epsilon, new sparse run fct

def run_simple_test(subset, copula="cvine", file_kld_appendage = "", evaluation_data_type="training_data", print_info=True):

    sub_kld_summary_statistics_training_data_list = []

    for index, row in subset.iterrows():
        if print_info == True:
            print("run_id",row["tags.mlflow.runName"])

        artifact_uri = row["artifact_uri"]
        seed = row["tags.seed"]
        train_obs = row["tags.seed"]
        copula_par=row["tags.copula_par"]
        train_obs=row["tags.train_obs"]
        uri = artifact_uri + "/precision_matrix_summary_statistics.csv"
        sub_pmatrix_summary_statistics_training_data = pd.read_csv(uri)

        
        family, par, rvinematrix, kendalltau = load_sim_infos(seed,
                        train_obs = train_obs,
                        copula = copula,
                        copula_par = copula_par)
        true_ci_graph = get_true_ci_graph(family, par, rvinematrix, kendalltau)
        
        with open(artifact_uri[7:]+"/actual_log_distribution_glq_list_{}{}.pkl".format(evaluation_data_type,file_kld_appendage), 'rb') as file:
            actual_log_distribution_glq_list = pickle.load(file)
        with open(artifact_uri[7:]+"/under_ci_assumption_log_distribution_glq_list_{}{}.pkl".format(evaluation_data_type,file_kld_appendage), 'rb') as file:
            under_ci_assumption_log_distribution_glq_list = pickle.load(file)
        
        sub_pmatrix_summary_statistics_training_data = sub_pmatrix_summary_statistics_training_data[["var_row", "var_col"]]
        
        # goes from zero to 5 in 0.1 steps
        for epsilon in range(0,21,1):
            epsilon = epsilon/20
            
            simpletest_pvalue_list = []
            for row_num in range(sub_pmatrix_summary_statistics_training_data.shape[0]):
                var_row_num = int(sub_pmatrix_summary_statistics_training_data.iloc[row_num]["var_row"])
                var_col_num = int(sub_pmatrix_summary_statistics_training_data.iloc[row_num]["var_col"])
                
                actual_log_distribution_glq = actual_log_distribution_glq_list[row_num]
                under_ci_assumption_log_distribution_glq = under_ci_assumption_log_distribution_glq_list[row_num]
                
                # simply need to remove the NaNs and Infs
                ll_dev = actual_log_distribution_glq - under_ci_assumption_log_distribution_glq
                bool_inf_nan = torch.isnan(ll_dev) | torch.isinf(ll_dev)
                actual_log_distribution_glq = actual_log_distribution_glq[~bool_inf_nan]
                under_ci_assumption_log_distribution_glq = under_ci_assumption_log_distribution_glq[~bool_inf_nan]
                
                #if sum(torch.isnan(actual_log_distribution_glq)) > 0:
                #    print("Percentage of NaN in Model", sum(torch.isnan(actual_log_distribution_glq))/actual_log_distribution_glq.size(0))
                #if sum(torch.isinf(actual_log_distribution_glq)) > 0:
                #    print("Percentage of Inf in Model", sum(torch.isinf(actual_log_distribution_glq))/actual_log_distribution_glq.size(0))
                #if sum(torch.isnan(under_ci_assumption_log_distribution_glq)) > 0:
                #    print("Percentage of NaN in CI-Model", sum(torch.isnan(under_ci_assumption_log_distribution_glq))/under_ci_assumption_log_distribution_glq.size(0))
                #if sum(torch.isinf(under_ci_assumption_log_distribution_glq)) > 0:
                #    print("Percentage of Inf in CI-Model", sum(torch.isinf(under_ci_assumption_log_distribution_glq))/under_ci_assumption_log_distribution_glq.size(0))
                
                d = torch.mean(actual_log_distribution_glq - under_ci_assumption_log_distribution_glq)
                #d = torch.mean(torch.exp(actual_log_distribution_glq) *(actual_log_distribution_glq - under_ci_assumption_log_distribution_glq))
                
                # if not even number of obs drop last one
                if actual_log_distribution_glq.size(0) % 2 != 0:
                    actual_log_distribution_glq = actual_log_distribution_glq[:-1]
                    under_ci_assumption_log_distribution_glq = under_ci_assumption_log_distribution_glq[:-1]
                
                # A Simple Parametric Test  Section 3 Formulas:
                N = len(actual_log_distribution_glq)
                
                #var_A = actual_log_distribution_glq.var()
                #var_B = under_ci_assumption_log_distribution_glq.var()
                #covar_AB = np.cov(actual_log_distribution_glq, under_ci_assumption_log_distribution_glq)[0,1]
                covar_matrix = np.cov(actual_log_distribution_glq, under_ci_assumption_log_distribution_glq)
                var_A = covar_matrix[0,0]
                var_B = covar_matrix[1,1]
                covar_AB = covar_matrix[0,1]
                
                var_joined = var_A + var_B - 2*covar_AB
                
                #w_i = torch.FloatTensor([1,1+epsilon]).repeat(round(actual_log_distribution_glq.size(0)/2))
                #w_i_plus_1 = torch.FloatTensor([1+epsilon,1]).repeat(round(actual_log_distribution_glq.size(0)/2))
                #d = (w_i * actual_log_distribution_glq - w_i_plus_1 * under_ci_assumption_log_distribution_glq).mean()
                
                var_d = (1 + epsilon) * var_joined + epsilon**2 / 2 * (var_A + var_B)
                
                #t = N**0.5 * d / var_d**0.5
                
                #d = torch.sum(actual_log_distribution_glq - under_ci_assumption_log_distribution_glq) / N
                d_split = torch.sum(actual_log_distribution_glq[list(range(0,N,2))] - under_ci_assumption_log_distribution_glq[list(range(1,N,2))]) / N
                
                t = N**0.5 * (d + epsilon*d_split) / var_d**0.5
                
                p = (1 - torch.distributions.Normal(0,1).cdf(t)).item()
                
                simpletest_pvalue_list.append(p)

            sub_pmatrix_summary_statistics_training_data["simpletest_pvalue"+"_epsilon_"+str(epsilon)] = simpletest_pvalue_list
            
        sub_pmatrix_summary_statistics_training_data = pd.merge(sub_pmatrix_summary_statistics_training_data, true_ci_graph, on=['var_row', 'var_col'])
                
                
        sub_kld_summary_statistics_training_data_list.append(sub_pmatrix_summary_statistics_training_data)
        
    return sub_kld_summary_statistics_training_data_list
