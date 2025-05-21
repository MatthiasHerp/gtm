import torch
import pandas as pd
import time
import pickle
import multiprocessing
import warnings
from gtm.gtm_plots_analysis.independence_kld_process_row import *
from gtm.gtm_plots_analysis.compute_precision_matrix_summary_statistics import *



def compute_conditional_independence_kld(self,
                                        y = None,
                                        x = False,
                                        evaluation_data_type = "data",
                                        num_processes=10,
                                        sample_size = 1000,
                                        num_points_quad=20,
                                        optimized=False,
                                        copula_only=False):
    
            
        # in case of gpu cuda compute
        if evaluation_data_type == "data":
            evaluation_data = y[:sample_size]  # Adjust this based on your needs
            if copula_only == True:
                evaluation_data = self.after_transformation(evaluation_data)
        elif evaluation_data_type == "uniform_random_samples":
            evaluation_data = torch.distributions.Uniform(-3,3).sample([sample_size, self.y_train.size(1)])
        elif evaluation_data_type == "samples_from_model":
            evaluation_data = self.sample(sample_size).detach()
            if copula_only == True:
                evaluation_data = self.after_transformation(evaluation_data)
                

        if copula_only == True:
            self.num_trans_layers = 0
        
        precision_matrix = self.compute_precision_matrix(evaluation_data).detach().cpu()
        correlation_matrix = self.compute_correlation_matrix(evaluation_data).detach().cpu()
        
        
        precision_matrix_summary_statistics = compute_precision_matrix_summary_statistics(precision_matrix)
        
        actual_log_distribution_glq_list = []
        under_ci_assumption_log_distribution_glq_list = []
        
        start = time.time()
        
        # Using Pool from the multiprocessing module
        if num_processes > 1:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.starmap(independence_kld_process_row, [(row_num, precision_matrix_summary_statistics, evaluation_data, self, num_points_quad, optimized) for row_num in range(precision_matrix_summary_statistics.shape[0])])
            # Unpacking the results
            actual_log_distribution_glq_list, under_ci_assumption_log_distribution_glq_list, under_ci_assumption_log_distribution_glq_full_data_list = zip(*results)
        else:
            results = [independence_kld_process_row(row_num, precision_matrix_summary_statistics, evaluation_data, self, num_points_quad, optimized) for row_num in range(precision_matrix_summary_statistics.shape[0])]
            
            actual_log_distribution_glq_list, under_ci_assumption_log_distribution_glq_list, under_ci_assumption_log_distribution_glq_full_data_list = zip(*results)
                
        end = time.time()
        
        print(f"Time taken: {end-start}")
    
        print("All rows processed.")
            
         
        precision_abs_mean_list = []
        precision_square_mean_list = []
        cond_correlation_abs_mean_list = []
        cond_correlation_square_mean_list = []
        kld_list = []
        iae_list = []

        for row_num in range(precision_matrix_summary_statistics.shape[0]):
            var_row_num = int(precision_matrix_summary_statistics.iloc[row_num]["var_row"])
            var_col_num = int(precision_matrix_summary_statistics.iloc[row_num]["var_col"])
            
            
            precision_abs_mean = precision_matrix[:,var_row_num,var_col_num].abs().mean()
            precision_square_mean = precision_matrix[:,var_row_num,var_col_num].square().mean()
            
            cond_correlation_abs_mean = correlation_matrix[:,var_row_num,var_col_num].abs().mean()
            cond_correlation_square_mean = correlation_matrix[:,var_row_num,var_col_num].square().mean()
            
            actual_log_distribution_glq = actual_log_distribution_glq_list[row_num]
            under_ci_assumption_log_distribution_glq = under_ci_assumption_log_distribution_glq_list[row_num]
            
            # in case of gpu cuda compute
            if evaluation_data_type == "data" or evaluation_data_type == "samples_from_model":
                ll_dev = actual_log_distribution_glq - under_ci_assumption_log_distribution_glq
                ll_dev = ll_dev[~torch.isnan(ll_dev)]
                ll_dev = ll_dev[~torch.isinf(ll_dev)]
                ll_dev = ll_dev[ll_dev.abs() < ll_dev.abs().quantile(0.98)]
                kld = ll_dev.mean()
                
                ll_dev2 = torch.abs(torch.exp(actual_log_distribution_glq) - torch.exp(under_ci_assumption_log_distribution_glq)) / torch.exp(actual_log_distribution_glq)
                ll_dev2 = ll_dev2[~torch.isnan(ll_dev2)]
                ll_dev2 = ll_dev2[~torch.isinf(ll_dev2)]
                ll_dev2 = ll_dev2[ll_dev2.abs() < ll_dev2.abs().quantile(0.98)]
                iae = ll_dev2.mean()
                
            elif evaluation_data_type == "uniform_random_samples":
                ll_dev = torch.exp(actual_log_distribution_glq) * (actual_log_distribution_glq - under_ci_assumption_log_distribution_glq)
                ll_dev = ll_dev[~torch.isnan(ll_dev)]
                ll_dev = ll_dev[~torch.isinf(ll_dev)]
                kld = ll_dev.mean()
                
                ll_dev2 = torch.abs(torch.exp(actual_log_distribution_glq) - torch.exp(under_ci_assumption_log_distribution_glq))
                ll_dev2 = ll_dev2[~torch.isnan(ll_dev2)]
                ll_dev2 = ll_dev2[~torch.isinf(ll_dev2)]
                iae = ll_dev2.mean()
            
            precision_abs_mean_list.append(precision_abs_mean.item())
            precision_square_mean_list.append(precision_square_mean.item())
            cond_correlation_abs_mean_list.append(cond_correlation_abs_mean.item())
            cond_correlation_square_mean_list.append(cond_correlation_square_mean.item())

            kld_list.append(kld.item())
            iae_list.append(iae.item())
            
            #print("Finished row_num: ", row_num)
            
        precision_matrix_summary_statistics["precision_abs_mean"] = precision_abs_mean_list
        precision_matrix_summary_statistics["precision_square_mean"] = precision_square_mean_list
        precision_matrix_summary_statistics["cond_correlation_abs_mean"] = cond_correlation_abs_mean_list
        precision_matrix_summary_statistics["cond_correlation_square_mean"] = cond_correlation_square_mean_list
        precision_matrix_summary_statistics["kld"] = kld_list
        precision_matrix_summary_statistics["iae"] = iae_list
            
        sub_kld_summary_statistics = precision_matrix_summary_statistics[['var_row','var_col', 
                                                                          'precision_abs_mean',
                                                                          'precision_square_mean',
                                                                          'cond_correlation_abs_mean', 
                                                                          'cond_correlation_square_mean',
                                                                          'kld',
                                                                          'iae']]
        
        if copula_only == True:
            self.num_trans_layers = 1
        
        return sub_kld_summary_statistics
            