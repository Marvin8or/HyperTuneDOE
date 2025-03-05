# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 19:58:15 2025

@author: gabri
"""
import numpy as np
import pandas as pd
import pyDOE3 as doe3
import doe_utils as du

# from sklearn.model_selection import train_test_split

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
    "Concrete compressive strength(MPa, megapascals) ": "concrete_compressive_strength",
}  # MPa

df = df.rename(columns=new_columns_names)

# %% Evaluation of model with default settings
factors = {
    "max_depth": {-1: 10, 1: 300},
    "min_samples_split": {-1: 10, 1: 100},
    "min_samples_leaf": {-1: 10, 1: 100},
    "min_weight_fraction_leaf": {-1: 0.1, 1: 0.4},
    "max_features": {-1: 2, 1: 6},
    "max_leaf_nodes": {-1: 10, 1: 100},
    "ccp_alpha": {-1: 0.1, 1: 0.2},
}

aliases = {
    "max_depth": "a",
    "min_samples_split": "b",
    "min_samples_leaf": "c",
    "min_weight_fraction_leaf": "d",
    "max_features": "e",
    "max_leaf_nodes": "f",
    "ccp_alpha": "g",
}
resolution_3_design_generator = "a b c d e ab cde"
design = doe3.fracfact(resolution_3_design_generator)

# design = doe3.fold(design)
print("\n".join(doe3.fracfact_aliasing(design)[0]))


design_df = pd.DataFrame(columns=list(factors.keys()), data=design)


X, y = (
    df[[c for c in df.columns if c != "concrete_compressive_strength"]],
    df["concrete_compressive_strength"],
)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=du.TSIZE, shuffle=True, random_state=du.RSTATE
# )


mapped_design = du.mapp_design_to_real_values(design, factors)
mapped_design_df = pd.DataFrame(mapped_design)
mapped_design_df[
    ["max_depth", "max_features", "max_leaf_nodes"]
] = mapped_design_df[["max_depth", "max_features", "max_leaf_nodes"]].replace(
    {np.nan: None}
)
results = du.run_experiments(
    X, y, target=du.TARGET, mapped_exp_des_template=mapped_design_df, n_reps=5
)
results = pd.concat([design_df, results], axis=1)


# %% ANOVA on ffd screening experiment


if False:
    du.create_main_effects_plots(
        results,
        du.TARGET,
        figsize=du.FIGSIZE_HALF,
        save_prefix="[1]",
        # save_path="C:/Users/gabri/OneDrive/Desktop/Dokumenti doktorat/MZIR/Seminarski rad/Images",
    )

_, anova_results = du.run_anova(
    results, factors, du.TARGET, n_interactions=2, anova_typ=2
)
anova_results = du.round_dataframe(anova_results)
anova_results_all = anova_results.copy()
anova_results = anova_results[anova_results["PR(>F)"] < 0.05]
best_combination = results.iloc[np.argmin(results["MSE"].values)]
worst_combination = results.iloc[np.argmax(results["MSE"].values)]
# =============================================================================
# Significant main effects
# min_weight_fraction_leaf, max_features, min_samples_leaf
# =============================================================================
# =============================================================================
# Significant interactions
# max_depth:ccp_alpha, min_samples_leaf:min_weight_fraction_leaf, min_samples_leaf:max_features
# min_weight_fraction_leaf:max_features
# =============================================================================
# %% Save results
if False:
    results_path = "C:/Projects/MZIR/experiments_results"
    results.to_excel(
        results_path + "/" + "[1] FracFactDesign_screening_results.xlsx"
    )
    anova_results_all.to_excel(
        results_path + "/" + "[1] FracFactDesign_ANOVA_results.xlsx"
    )
    anova_results.to_excel(
        results_path
        + "/"
        + "[1] FracFactDesign_ANOVA_results_significant.xlsx"
    )
