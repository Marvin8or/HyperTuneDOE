# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:43:56 2025

@author: gabri
"""
import numpy as np
import pandas as pd
import pyDOE3 as doe3
import doe_utils as du
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

file_path = "C:/Projects/MZIR/datasets/Concrete_Data.xls"

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)
new_columns_names = {
    "Cement (component 1)(kg in a m^3 mixture)": "cement",  # kg in a m^3 mixture
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "blast_furnace_slag",  # kg in a m^3 mixture
    "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash",  # kg in a m^3 mixture
    "Water  (component 4)(kg in a m^3 mixture)": "water",  # kg in a m^3 mixture
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",  # kg in a m^3 mixture
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_aggregate",  # kg in a m^3 mixture
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine_aggregate",  # kg in a m^3 mixture
    "Age (day)": "age",  # day
    "Concrete compressive strength(MPa, megapascals) ": "concrete_compressive_strength",  # MPa
}

df = df.rename(columns=new_columns_names)
# %% Evaluation of model with default settings with full factorial design
# =============================================================================
# Significant main effects
# max_depth, min_weight_fraction_leaf, max_features
# =============================================================================
factors = {
    "max_depth": {-1: 2, 1: 10},
    "min_weight_fraction_leaf": {-1: 0, 1: 0.2},
    "max_features": {-1: 1, 1: 8},
}

design = doe3.ff2n(len(factors.keys()))
print("\n".join(doe3.fracfact_aliasing(design)[0]))

design_df = pd.DataFrame(columns=list(factors.keys()), data=design)

mapped_design = du.mapp_design_to_real_values(design, factors)
mapped_design_df = pd.DataFrame(mapped_design)

X, y = (
    df[[c for c in df.columns if c != "concrete_compressive_strength"]],
    df["concrete_compressive_strength"],
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=du.TSIZE, shuffle=True, random_state=du.RSTATE
)

results = du.run_experiments(
    X,
    y,
    target=du.TARGET,
    mapped_exp_des_template=mapped_design_df,
    n_repetitions=3,
)
results = pd.concat([design_df, results], axis=1)
mapped_results = pd.concat([mapped_design_df, results], axis=1)


# %% ANOVA on full factorial

if True:
    du.create_main_effects_plots(results, du.TARGET, [du.TARGET])
    du.create_interaction_plots(results, du.TARGET, [du.TARGET])

anova_results = du.run_anova(
    results, factors, du.TARGET, n_interactions=2, anova_typ=2
)

# %% Save results
if False:
    results_path = "C:/Projects/MZIR/experiments_results"
    results.to_excel(
        results_path + "/" + "[3] FullFactDesign_screening_results.xlsx"
    )
    anova_results.to_excel(
        results_path + "/" + "[3] FullFactDesign_ANOVA_results.xlsx"
    )
