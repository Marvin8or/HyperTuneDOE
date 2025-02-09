# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 19:58:15 2025

@author: gabri
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyDOE3 as doe

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

df.rename(columns=new_columns_names, inplace=True)

# %% Evaluation of model with default settings
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

RSTATE = 11
TSIZE = 0.2
NSPLITS = 10


def evaluate_model(X, y, target, model_hyperparams=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TSIZE, shuffle=True, random_state=RSTATE
    )

    kf = KFold(n_splits=NSPLITS, shuffle=True)
    kf_performance_metrics = {"MSE": [], "RMSE": [], "R2": []}

    if model_hyperparams is None:
        dtr_model = DecisionTreeRegressor(random_state=RSTATE)
    else:
        dtr_model = DecisionTreeRegressor(
            random_state=RSTATE, **model_hyperparams
        )

    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # Evaluating on split
        Xtrain, Xval = X_train.values[train_idx], X_train.values[val_idx]
        ytrain, yval = y_train.values[train_idx], y_train.values[val_idx]
        fitted_model = dtr_model.fit(Xtrain, ytrain.ravel())
        yvalpred = fitted_model.predict(Xval)

        kf_performance_metrics["MSE"].append(
            mean_squared_error(yval, yvalpred)
        )
        kf_performance_metrics["RMSE"].append(
            np.sqrt(mean_squared_error(yval, yvalpred))
        )
        kf_performance_metrics["R2"].append(r2_score(yval, yvalpred))

    mean_target_over_folds = np.round(
        np.mean(kf_performance_metrics[target]), 3
    )
    return mean_target_over_folds


factors = {
    "max_depth": [2, 20],
    "min_samples_split": [2, 50],
    "min_samples_leaf": [1, 20],
    "min_weight_fraction_leaf": [0, 0.2],
    "max_features": [1, 8],
    "max_leaf_nodes": [100, None],
    "ccp_alpha": [0, 0.2],
}

factors = {
    "max_depth": {-1: 2, 1: 20},
    "min_samples_split": {-1: 2, 1: 50},
    "min_samples_leaf": {-1: 1, 1: 20},
    "min_weight_fraction_leaf": {-1: 0, 1: 0.2},
    "max_features": {-1: 1, 1: 8},
    "max_leaf_nodes": {-1: 100, 1: None},
    "ccp_alpha": {-1: 0, 1: 0.2},
}

MAX_DEPTH = "a"
MIN_SAMPLES_SPLIT = "b"
MIN_SAMPLES_LEAF = "c"
MIN_WEIGHT_FRACTION_LEAF = "d"
MAX_FEATURES = "e"
MAX_LEAF_NODES = "f"
CCP_ALPHA = "g"

design, alias_map, alias_cost = doe.fracfact_opt(7, 2)
resolution_3_design = "a b c d e ab ac"

print("\n".join(doe.fracfact_aliasing(doe.fracfact(resolution_3_design))[0]))
# print(design)
ff_exp_design = doe.fracfact(resolution_3_design)

# print(ff_exp_design)

mapped_ff_exp_design = {}
for ci, k in enumerate(factors.keys()):
    mapped_ff_exp_design[k] = []
    for ri in range(len(ff_exp_design)):
        mapped_ff_exp_design[k].append(factors[k][int(ff_exp_design[ri][ci])])


mapped_ff_exp_design_df = pd.DataFrame(mapped_ff_exp_design)
mapped_ff_exp_design_df.max_leaf_nodes = (
    mapped_ff_exp_design_df.max_leaf_nodes.replace(np.nan, None)
)

ff_exp_design_df = pd.DataFrame(
    columns=list(factors.keys()), data=ff_exp_design
)


def run_experiments(X, y, target, mapped_exp_des_template, n_repetitions=2):
    results_df = pd.DataFrame()
    for repetition_idx in range(n_repetitions):
        rows = mapped_exp_des_template.to_dict(orient="records")
        repetition_result = np.zeros(shape=(len(mapped_exp_des_template),))
        for i, run_hyperparams in enumerate(rows):
            if "max_leaf_nodes" in run_hyperparams.keys():
                if run_hyperparams["max_leaf_nodes"] is not None:
                    run_hyperparams["max_leaf_nodes"] = int(
                        run_hyperparams["max_leaf_nodes"]
                    )
            repetition_result[i] = evaluate_model(
                X, y, target, run_hyperparams
            )

        results_df[f"{target}_{repetition_idx+1}"] = repetition_result

    return results_df


X, y = (
    df[[c for c in df.columns if c != "concrete_compressive_strength"]],
    df["concrete_compressive_strength"],
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TSIZE, shuffle=True, random_state=RSTATE
)


results = run_experiments(
    X,
    y,
    target="MSE",
    mapped_exp_des_template=mapped_ff_exp_design_df,
    n_repetitions=1,
)
ffd_results = pd.concat([ff_exp_design_df, results], axis=1)
ffd_screening_results = pd.concat([mapped_ff_exp_design_df, results], axis=1)

results_path = "C:/Projects/MZIR/experiments_results"
ffd_screening_filename = "FracFactDesign_screening_results.xlsx"

if True:
    ffd_screening_results.to_excel(results_path + "/" + ffd_screening_filename)

# %% ANOVA on ffd screening experiment
import seaborn as sns
from itertools import combinations
import statsmodels.api as sm
from statsmodels.formula.api import ols


def create_main_effects_plots(data, y, repetition_results: list):
    # Get all column names except for the response variable
    factors = [col for col in data.columns if col not in repetition_results]

    for fi, fname in enumerate(factors):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle(f"Main Effects Plot repetition_num: {y}")
        sns.pointplot(x=fname, y=y, data=data, ax=axs)
        axs.set_xlabel(fname)
        axs.set_ylabel(y)
        axs.grid()
        axs.set_ylim(np.min(data[y]), np.max(data[y]))

    plt.tight_layout()


# 'Filtration_rate ~ T + CoF + RPM + T:CoF + T:RPM'
def generate_anova_formula(response_name, factor_names, max_interactions):
    if max_interactions <= 0 or max_interactions > len(factor_names):
        raise ValueError(
            "'max_interactions' must be in range 0 < x <= len('factor_names')"
        )

    model_formula = ""
    model_formula += response_name + " ~ "
    model_formula += " + ".join(factor_names)

    combined_factors_at_interaction_num = []
    for interaction_num in range(1, max_interactions + 1):
        for combo in combinations(factor_names, interaction_num):
            combined_factors_at_interaction_num.append(":".join(combo))

    model_formula += " + " + " + ".join(combined_factors_at_interaction_num)
    return model_formula


if False:
    create_main_effects_plots(ffd_results, "MSE_1", ["MSE_1"])

anova_model_formula = generate_anova_formula(
    "MSE_1", factors.keys(), max_interactions=2
)
model = ols(anova_model_formula, data=ffd_results).fit()
anova_results_1 = sm.stats.anova_lm(model, typ=2)
print(anova_results_1)

MAX_DEPTH = "a"
MIN_SAMPLES_SPLIT = "b"
MIN_SAMPLES_LEAF = "c"
MIN_WEIGHT_FRACTION_LEAF = "d"
MAX_FEATURES = "e"
MAX_LEAF_NODES = "f"
CCP_ALPHA = "g"

if False:
    anova_results_1.to_excel(
        results_path + "/" + "FracFactDesign_ANOVA_results_1.xlsx"
    )

# %% De alias main effects

folded_ff_exp_design = doe.fold(ff_exp_design)

mapped_folded_ff_exp_design = {}
for ci, k in enumerate(factors.keys()):
    mapped_folded_ff_exp_design[k] = []
    for ri in range(len(folded_ff_exp_design)):
        mapped_folded_ff_exp_design[k].append(
            factors[k][int(folded_ff_exp_design[ri][ci])]
        )


mapped_folded_ff_exp_design_df = pd.DataFrame(mapped_folded_ff_exp_design)
mapped_folded_ff_exp_design_df.max_leaf_nodes = (
    mapped_folded_ff_exp_design_df.max_leaf_nodes.replace(np.nan, None)
)

folded_ff_exp_design_df = pd.DataFrame(
    columns=list(factors.keys()), data=folded_ff_exp_design
)

# %% Run experiments on folded design
folded_results = run_experiments(
    X,
    y,
    target="MSE",
    mapped_exp_des_template=mapped_folded_ff_exp_design_df,
    n_repetitions=1,
)
folded_ffd_results = pd.concat(
    [folded_ff_exp_design_df, folded_results], axis=1
)
folded_ffd_screening_results = pd.concat(
    [mapped_folded_ff_exp_design_df, folded_results], axis=1
)


if False:
    folded_ffd_screening_results.to_excel(
        results_path + "/" + "FracFactDesign_folded_screening_results.xlsx"
    )

# %% ANOVA on folded ffd screening experiment
# anova_model_formula = generate_anova_formula("MSE_1", factors.keys(), max_interactions=2)
model = ols(anova_model_formula, data=folded_ffd_results).fit()
anova_results_2 = sm.stats.anova_lm(model, typ=2)
print(anova_results_2)

if False:
    anova_results_2.to_excel(
        results_path + "/" + "FracFactDesign_ANOVA_results_2.xlsx"
    )

# %% Full factorial of destilled factors
# max_Depth, min_weight_fraction_leaf, max_features, min_samples_leaf


def create_interaction_plots(data, y):
    # Get all column names except for the response variable
    factors = [col for col in data.columns if col != y]

    # Generate all possible pairs of factors
    factor_pairs = combinations(factors, 2)

    for f1, f2 in factor_pairs:
        levels_f1 = np.unique(data[f1])
        levels_f2 = np.unique(data[f2])

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        for l2 in levels_f2:
            means = []
            for l1 in levels_f1:
                mask = (data[f1] == l1) & (data[f2] == l2)
                means.append(np.mean(data[y][mask]))
            axs.plot(levels_f1, means, marker="o", label=f"{f2}={int(l2)}")

        axs.set_xlabel(f1)
        axs.set_ylabel(f"Mean {y}")
        axs.set_ylim(np.min(data[y]), np.max(data[y]))
        fig.suptitle(f"{f1} : {f2}")
        axs.legend()
        axs.grid(True, linestyle="--", alpha=0.6)


significant_effects = ["max_depth", "min_weight_fraction_leaf", "max_features"]

full_factorial_design = doe.ff2n(len(significant_effects))

mapped_full_factorial_design = {}
for ci, k in enumerate(significant_effects):
    mapped_full_factorial_design[k] = []
    for ri in range(len(full_factorial_design)):
        mapped_full_factorial_design[k].append(
            factors[k][int(full_factorial_design[ri][ci])]
        )

mapped_ffd_df = pd.DataFrame(
    columns=significant_effects, data=mapped_full_factorial_design
)

ffd_results = run_experiments(
    X, y, target="MSE", mapped_exp_des_template=mapped_ffd_df, n_repetitions=1
)

ffd_df = pd.concat(
    [
        pd.DataFrame(columns=significant_effects, data=full_factorial_design),
        ffd_results,
    ],
    axis=1,
)

if True:
    create_main_effects_plots(ffd_df, "MSE_1", ["MSE_1"])
    create_interaction_plots(ffd_df, "MSE_1")

# %% ANOVA on full factorial
anova_model_formula = generate_anova_formula("MSE_1", significant_effects, 2)
model = ols(anova_model_formula, data=ffd_df).fit()
anova_results_3 = sm.stats.anova_lm(model, typ=2)
print(anova_results_3)

if False:
    anova_results_3.to_excel(
        results_path + "/" + "FracFactDesign_ANOVA_results_3.xlsx"
    )
