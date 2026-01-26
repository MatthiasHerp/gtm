import mlflow
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


def get_data_original_and_bootstrap(df_runs,data="training_and_validation"):
    
    N_train = df_runs['tags.N_train'].astype(float).iloc[0].item()
    N_validate = df_runs['tags.N_validate'].astype(float).iloc[0].item()
    N_train_proportion = float(N_train / (N_train + N_validate))
    N_validate_proportion = float(N_validate / (N_train + N_validate))

    table_diffs_list = []
    table_original_dict = {}
    for seed in df_runs['tags.seed_value'].unique():
        
        df_of_seed = df_runs[df_runs['tags.seed_value']==str(seed)]
        
        # TEMPORARY: for future bootstrap_warpspeed is a tag for now out of name
        if "tags.bootstrap_warpspeed" in df_of_seed.columns:
            bootstrap_bool = df_of_seed["tags.bootstrap_warpspeed"].str.contains("True")
        else:
            bootstrap_bool = df_of_seed["tags.mlflow.runName"].str.contains("bootstrap")
        df_run_bootstrap = df_of_seed[bootstrap_bool]
        df_run_original = df_of_seed[~bootstrap_bool]

        if data=="training_and_validation":
            # for train data and validate data
            #artifact_path = "conditional_independence_table_model_samples.csv" #"conditional_independence_table_data_train.csv"
            artifact_path = "conditional_independence_table_data_train.csv"
            table_bootstrap_train = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=df_run_bootstrap['run_id'].item(), artifact_path=artifact_path))
            table_original_train = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=df_run_original['run_id'].item(), artifact_path=artifact_path))
            #artifact_path = "conditional_independence_table_model_samples.csv" #"conditional_independence_table_data_validate.csv"
            artifact_path = "conditional_independence_table_data_validate.csv"
            table_bootstrap_validate = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=df_run_bootstrap['run_id'].item(), artifact_path=artifact_path))
            table_original_validate = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=df_run_original['run_id'].item(), artifact_path=artifact_path))
            
            # weight results together so its same data as used for training with GTM
            table_original = table_original_train.copy()
            table_bootstrap = table_bootstrap_train.copy()
            
            # check same ordering of var_row and var_col
            table_original.sort_values(by=['var_row', 'var_col'], inplace=True)
            table_bootstrap.sort_values(by=['var_row', 'var_col'], inplace=True)
            
            table_original.iloc[:, :3] = table_original.iloc[:, :3].astype(float)
            table_bootstrap.iloc[:, :3] = table_bootstrap.iloc[:, :3].astype(float)
            
            table_original.iloc[:, :3] = table_original_train.iloc[:, :3].astype(float) * N_train_proportion + table_original_validate.iloc[:, :3].astype(float) * N_validate_proportion
            table_bootstrap.iloc[:, :3] = table_bootstrap_train.iloc[:, :3].astype(float) * N_train_proportion + table_bootstrap_validate.iloc[:, :3].astype(float) * N_validate_proportion
        elif data=="synthetic":
            # for train data and validate data
            artifact_path = "conditional_independence_table_model_samples.csv" 
            table_bootstrap = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=df_run_bootstrap['run_id'].item(), artifact_path=artifact_path))
            table_original = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=df_run_original['run_id'].item(), artifact_path=artifact_path))
            
            # check same ordering of var_row and var_col
            table_original.sort_values(by=['var_row', 'var_col'], inplace=True)
            table_bootstrap.sort_values(by=['var_row', 'var_col'], inplace=True)
            
        # compute difference in tables for all except first 3 columns and last 3, laster are infors of dependence, tau of copula and copula
        table_diff = table_original.copy()
        table_diff.iloc[:, 3:9] = table_original.iloc[:, 3:9] - table_bootstrap.iloc[:, 3:9] 
            
        # across all seeds store the differences table in a joint array
        table_diffs_list.append(table_diff)
        table_original_dict[seed] = table_original
        
    return table_diffs_list, table_original_dict


def plot_metric_percentage_above_zero_bootstrap_intervals_across_seeds(df_runs,table_original_dict,table_diffs_list,df_structure,metric,ylim=(-0.1,1.1)):
    
    df_structure.sort_values(by=['var_row', 'var_col'], inplace=True)
    dependence = df_structure["dependence"]
    #if dependence is 1 set to lightblue else to red
    colors = ['lightblue' if dep == 1 else 'red' for dep in dependence]
    
    first_table_original_dict = table_original_dict[next(iter(table_original_dict))]
    
    dict_diffs = {}
    for pair in zip(first_table_original_dict['var_row'], first_table_original_dict['var_col']):
        varrow = pair[0]
        varcol = pair[1]
        diffs_to_plot = []
        for table_diffs in table_diffs_list:
            diffs_value = table_diffs[(table_diffs['var_col'] == varcol) & (table_diffs['var_row'] == varrow)][metric].values[0]
            diffs_to_plot.append(diffs_value)
            
        dict_diffs[f"{int(varrow)}-{int(varcol)}"] = diffs_to_plot
    
    # for each seed plot boxplot of differences around orginal kld
    list_percentage_ci_above_0 = []
    for seed in df_runs['tags.seed_value'].unique():
        table_original = table_original_dict[seed]

        # add vector to the 
        centered_data = pd.DataFrame(dict_diffs) + np.array(table_original[metric])
        
        percentage_ci_above_0 = (centered_data > 0).mean(0)
        
        list_percentage_ci_above_0.append(percentage_ci_above_0)

    df_percentage_ci_above_0 = pd.DataFrame(list_percentage_ci_above_0)

    # --- Boxplot with 90% interval (whis=[5, 95]) ---
    sns.boxplot(data=df_percentage_ci_above_0, whis=[5, 95], showfliers=False, palette=colors)
    # Colors on independence from above
    sns.stripplot(data=df_percentage_ci_above_0, palette=colors, alpha=0.4, edgecolor="black", dodge=True, jitter=0.25)

    plt.ylim(ylim)
    plt.axhline(0, color='red', linestyle='--')
    plt.ylabel("Percentage")
    plt.title("Boxplot of Percentage of the Bootstrap Interal Points above 0 {} per Pair across Seeds".format(metric))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
def plot_metric_values_across_seeds(df_runs,table_original_dict,df_structure,metric,ylim=(-0.05,0.25)):
    
    df_structure.sort_values(by=['var_row', 'var_col'], inplace=True)
    dependence = df_structure["dependence"]
    #if dependence is 1 set to lightblue else to red
    colors = ['lightblue' if dep == 1 else 'red' for dep in dependence]
    
    list_average_ci = []
    for seed in df_runs['tags.seed_value'].unique():
        table_original = table_original_dict[seed]

        # add vector to the 
        centered_data = np.array(table_original[metric]) #pd.DataFrame(dict_diffs) + 
        
        average_ci = centered_data#.mean(0)

        list_average_ci.append(average_ci)

    df_average_ci = pd.DataFrame(list_average_ci)
    # --- Boxplot with 90% interval (whis=[5, 95]) ---
    sns.boxplot(data=df_average_ci, whis=[5, 95], showfliers=False, palette=colors)
    # Colors on independence from above
    sns.stripplot(data=df_average_ci, palette=colors, alpha=0.4, edgecolor="black", dodge=True, jitter=0.25)

    plt.ylim(ylim)
    plt.axhline(0, color='red', linestyle='--')
    plt.ylabel("Value")
    plt.title("Boxplot of {}  per Pair across Seeds".format(metric))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
def plot_metric_bootstrap_intervals_for_each_seed(df_runs,table_original_dict,table_diffs_list,df_structure,metric,ylim=(-0.05,0.25)):
    
    df_structure.sort_values(by=['var_row', 'var_col'], inplace=True)
    dependence = df_structure["dependence"]
    #if dependence is 1 set to lightblue else to red
    colors = ['lightblue' if dep == 1 else 'red' for dep in dependence]
    
    first_table_original_dict = table_original_dict[next(iter(table_original_dict))]
    
    dict_diffs = {}
    for pair in zip(first_table_original_dict['var_row'], first_table_original_dict['var_col']):
        varrow = pair[0]
        varcol = pair[1]
        diffs_to_plot = []
        for table_diffs in table_diffs_list:
            diffs_value = table_diffs[(table_diffs['var_col'] == varcol) & (table_diffs['var_row'] == varrow)][metric].values[0]
            diffs_to_plot.append(diffs_value)
            
        dict_diffs[f"{int(varrow)}-{int(varcol)}"] = diffs_to_plot
        

    for seed in df_runs['tags.seed_value'].unique():
        table_original = table_original_dict[seed]

        # add vector to the 
        centered_data = pd.DataFrame(dict_diffs) + np.array(table_original[metric])

        # check that the centering worked for all columns
        if all((pd.DataFrame(dict_diffs).iloc[:,0] + np.array(table_original[metric])[0]) == centered_data.iloc[:,0]):
            pass
        else:
            print("Centering failed.")
            
        if all((pd.DataFrame(dict_diffs).iloc[:,3] + np.array(table_original[metric])[3]) == centered_data.iloc[:,3]):
            pass
        else:
            print("Centering failed.")

        plt.figure(figsize=(12, 6))
        
        
        #sns.violinplot(data=centered_data)
        
        # --- Boxplot with 90% interval (whis=[5, 95]) ---
        sns.boxplot(data=centered_data, whis=[5, 95], showfliers=False, palette=colors)
        # Colors on independence from above

        # --- Show data points overlaid (jittered) ---
        # Use stripplot for point overlay; set jitter for clarity
        sns.stripplot(data=centered_data, palette=colors, alpha=0.4, edgecolor="black", dodge=True, jitter=0.25)
        
        
        plt.axhline(0, color='red', linestyle='--')
        plt.ylabel("Value")
        plt.title("{} Warpspeed Bootstrap Intervals around Original, Seed ".format(metric) + str(seed))
        plt.xticks(rotation=45)
        plt.ylim(ylim)
        plt.tight_layout()
        plt.show()