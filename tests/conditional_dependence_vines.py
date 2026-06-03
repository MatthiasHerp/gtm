from gtm import *

# Sample Copulas Package
import pyvinecopulib as pv
# Helpers to generate specfific vines and analyze there conditional independencies
from demos.pyvinecopulib_simulation_helpers import *

# Other Stuff
import numpy as np
import torch
from torch.utils.data import DataLoader
from demos.dataset_helpers import Generic_Dataset
from sklearn.metrics import roc_auc_score, roc_curve


def test_general(D = 10,Independence_tree = 3):
    
    rvine_structure = pv.RVineStructure.simulate(D, seeds=[1])
    pair_copulas = sample_random_pair_copulas(D, Independence_tree=Independence_tree)
    vine_model = pv.Vinecop.from_structure(structure=rvine_structure, pair_copulas=pair_copulas)
    df = compute_conditional_dependence_table(vine_model, Independence_tree=Independence_tree)
    
        
    df_true_structure = df[["tree","edge","conditioned variables", "conditioned variable 1", "conditioned variable 2", "dependence", "var_row", "var_col"]]
    df_true_structure_sub = df_true_structure[["var_row", "var_col", "dependence"]]
    df_true_structure_sub.loc[:, "var_row"] = df_true_structure_sub["var_row"] - 1
    df_true_structure_sub.loc[:, "var_col"] = df_true_structure_sub["var_col"] - 1
    
    
    num_pairs = D*(D-1)/2
    
    # through error if false
    if (df_true_structure_sub["dependence"].mean() - sum([D-1-i for i in range(Independence_tree)])/num_pairs) != 0:
        print(df_true_structure_sub["dependence"].mean())
        print(sum([D-1-i for i in range(Independence_tree)])/num_pairs)
        raise ValueError(f"For {D} dimensions and {Independence_tree} independent trees: Error in the conditional dependence table. The number of dependent edges does not match the expected number given the Independence_tree parameter.")
    
    for i in range(D-1):
        for j in range(D-1-i):
            if i < Independence_tree:
                if pair_copulas[i][j].family.name == "indep":
                    raise ValueError(f"For {D} dimensions and {Independence_tree} independent trees: Error: Copula family is independent, but should be dependent in trees till Independence_tree.")
            else:
                if pair_copulas[i][j].family.name != "indep":
                    raise ValueError(f"For {D} dimensions and {Independence_tree} independent trees: Error: Copula family is not independent, but should be independent in trees after Independence_tree.")


if __name__ == "__main__":
    set_seeds(1)
    
    test_general(D = 10,Independence_tree = 3)
    
    test_general(D = 10,Independence_tree = 2)
    
    test_general(D = 10,Independence_tree = 1)
    
    test_general(D = 100,Independence_tree = 3)
    
    test_general(D = 100,Independence_tree = 2)
    
    print("no prints then it passed all tests")


