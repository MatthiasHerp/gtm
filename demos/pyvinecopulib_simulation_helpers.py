# GTM Package including GTM and the plot functions
import random

import matplotlib.pyplot as plt
import numpy as np

# Sample Copulas Package
import pyvinecopulib as pv
import seaborn as sns

# Other Stuff
import torch
from torch.utils.data import DataLoader, Dataset

from gtm import *


def set_seeds(seed_int):
    # Reproducibility
    # Infos from here: https://pytorch.org/docs/stable/notes/randomness.html
    # Set Seeds for Torch, Numpy and Python
    torch.manual_seed(seed_int)
    np.random.seed(seed_int)
    random.seed(seed_int)


def sample_random_pair_copulas(D, Independence_tree=2):
    pair_copulas = []

    # List of families to sample from (can add/remove)
    families = [
        # pv.BicopFamily.indep,
        pv.BicopFamily.gaussian,
        pv.BicopFamily.student,
        pv.BicopFamily.clayton,
        pv.BicopFamily.gumbel,
        pv.BicopFamily.frank,
        pv.BicopFamily.joe,
    ]

    # For each tree in the vine
    for tree in range(D - 1):
        bicop_list = []
        num_edges = D - tree - 1
        if tree >= Independence_tree:
            for _ in range(num_edges):
                bicop = pv.Bicop(family=pv.BicopFamily.indep, rotation=0)
                bicop_list.append(bicop)
            pair_copulas.append(bicop_list)
        else:
            for _ in range(num_edges):
                fam = np.random.choice(families)

                tau = 0.3 + 0.4 * np.random.random()
                neg = np.random.choice([-1, 1])

                # Sample rotation 0-3 for asymmetric families (e.g., Clayton, Gumbel, Frank)
                if fam in [
                    pv.BicopFamily.clayton,
                    pv.BicopFamily.gumbel,
                    pv.BicopFamily.joe,
                ]:
                    rotation = np.random.choice([0, 90, 180, 270])
                else:
                    rotation = 0
                if fam == pv.BicopFamily.student:
                    bicop = pv.Bicop(
                        family=fam,
                        rotation=rotation,
                        parameters=np.stack([np.array([neg * tau]), np.array([2])]),
                    )
                elif fam == pv.BicopFamily.gaussian:
                    bicop = pv.Bicop(
                        family=fam,
                        rotation=rotation,
                        parameters=np.stack([np.array([neg * tau])]),
                    )
                elif fam == pv.BicopFamily.frank:
                    bicop = pv.Bicop(family=fam, rotation=rotation)
                    bicop = pv.Bicop(
                        family=fam,
                        rotation=rotation,
                        parameters=bicop.tau_to_parameters(
                            np.stack([np.array([neg * tau])])
                        ),
                    )
                else:
                    bicop = pv.Bicop(family=fam, rotation=rotation)
                    bicop = pv.Bicop(
                        family=fam,
                        rotation=rotation,
                        parameters=bicop.tau_to_parameters(np.array([tau])),
                    )

                bicop_list.append(bicop)
            pair_copulas.append(bicop_list)

    return pair_copulas


# Usage example:
# rvine_structure should be created or loaded beforehand
# pair_copulas = sample_random_pair_copulas(rvine_structure)
# vine = pv.Vinecop(structure=rvine_structure, pair_copulas=pair_copulas)


import re

import pandas as pd


def compute_conditional_dependence_table(vine_model):
    text = vine_model.format()

    # Strip first line (title) and get all data lines
    lines = text.strip().split("\n")[2:]

    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([\d, ]+)\s+([\d, ]+)\s+([cI, ]+)\s+(\w+)\s+(-?\d+)?\s+(.+?)\s+([-\d.]+)?\s+([-\d.]+)?\s*$"
    )

    rows = []
    for line in lines:
        match = pattern.match(line)
        if match:
            rows.append(match.groups())
        else:
            print(f"Failed to parse line: {line}")

    columns = [
        "tree",
        "edge",
        "conditioned variables",
        "conditioning variables",
        "var_types",
        "family",
        "rotation",
        "parameters",
        "df",
        "tau",
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Optional: convert numeric columns
    for col in ["tree", "edge", "rotation", "df", "tau"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def split_variables(row):
        # Split by comma and strip spaces
        row["conditioned variables"] = row["conditioned variables"].replace(",", " ")
        nums = [x.strip() for x in row["conditioned variables"].split(" ")]
        if len(nums) >= 2:
            conditioned = " ".join(nums[:3])
            conditioning = " ".join(nums[3:]) if len(nums) > 2 else ""
        else:
            conditioned = row["conditioned variables"]
            conditioning = ""
        return pd.Series([conditioned, conditioning])

    # Apply the function to split the variable columns
    df[["conditioned variables", "conditioning variables"]] = df.apply(
        split_variables, axis=1
    )

    df["conditioned variable 1"] = [
        df["conditioned variables"][i].split(" ")[0] for i in range(df.shape[0])
    ]
    df["conditioned variable 1"] = df["conditioned variable 1"].astype(int)
    df["conditioned variable 2"] = [
        df["conditioned variables"][i].split(" ")[2] for i in range(df.shape[0])
    ]
    df["conditioned variable 2"] = df["conditioned variable 2"].astype(int)
    df["var_row"] = [
        (
            df["conditioned variable 1"][i]
            if df["conditioned variable 1"][i] > df["conditioned variable 2"][i]
            else df["conditioned variable 2"][i]
        )
        for i in range(df.shape[0])
    ]
    df["var_col"] = [
        (
            df["conditioned variable 2"][i]
            if df["conditioned variable 1"][i] > df["conditioned variable 2"][i]
            else df["conditioned variable 1"][i]
        )
        for i in range(df.shape[0])
    ]
    df["dependence"] = [1 if df["tree"][i] < 3 else 0 for i in range(df.shape[0])]

    return df
