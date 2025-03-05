# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:30:49 2025

@author: gabri
"""
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from itertools import combinations
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

RSTATE = 11
TSIZE = 0.2
NSPLITS = 3
TARGET = "MSE"
FIGSIZE_FULL = (10, 5.625)
FIGSIZE_HALF = (5.625, 5.625)
FONTSIZE_AXES = 12
FONTSIZE_TITLE = 15


def round_dataframe(df):
    def round_value1(x):
        return round(x, 3) if x >= 0.05 else x  # Keep original if < 0.05

    return df.map(round_value1)


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
        Xtrain, Xval = X_train.values[train_idx], X_train.values[val_idx]
        ytrain, yval = y_train.values[train_idx], y_train.values[val_idx]
        fitted_model = dtr_model.fit(Xtrain, ytrain.ravel())
        # print(fitted_model.get_params())
        yvalpred = fitted_model.predict(Xval)

        kf_performance_metrics["MSE"].append(
            mean_squared_error(yval, yvalpred)
        )
        kf_performance_metrics["R2"].append(r2_score(yval, yvalpred))

    mean_target_over_folds = np.round(
        np.mean(kf_performance_metrics[target]), 3
    )

    return mean_target_over_folds


def run_experiments(
    X, y, target, mapped_exp_des_template, n_reps, fixed_params=None
):
    results_df = pd.DataFrame()
    rows = mapped_exp_des_template.to_dict(orient="records")
    repetition_result = np.zeros(shape=(len(mapped_exp_des_template)))
    for i, run_hyperparams in enumerate(rows):
        if (
            "max_features" in run_hyperparams.keys()
            and run_hyperparams["max_features"] is not None
        ):
            run_hyperparams["max_features"] = int(
                run_hyperparams["max_features"]
            )
        if (
            "max_depth" in run_hyperparams.keys()
            and run_hyperparams["max_depth"] is not None
        ):
            run_hyperparams["max_depth"] = int(run_hyperparams["max_depth"])

        if (
            "max_leaf_nodes" in run_hyperparams.keys()
            and run_hyperparams["max_leaf_nodes"] is not None
        ):
            run_hyperparams["max_leaf_nodes"] = int(
                run_hyperparams["max_leaf_nodes"]
            )

        if fixed_params is not None:
            run_hyperparams = dict(**run_hyperparams, **fixed_params)

        for r in range(n_reps):
            repetition_result[i] += evaluate_model(
                X, y, target, run_hyperparams
            )
        repetition_result[i] = repetition_result[i] / n_reps

    results_df[f"{target}"] = repetition_result

    return results_df


def create_main_effects_plots(
    data, y, figsize=FIGSIZE_FULL, save_path=None, save_prefix=None
):
    # Get all column names except for the response variable
    factors = [col for col in data.columns if col != y]

    # Get levels
    levels = []
    for fi, fname in enumerate(factors):
        levels.append(np.unique(data[factors].values))

    levels = np.unique(levels)
    for fi, fname in enumerate(factors):
        means = []
        # print(levels)
        min_vals = []
        max_vals = []
        for l in levels:
            mask = data[fname] == l
            mean_val = np.mean(data[y][mask])
            means.append(mean_val)
            min_vals.append(np.abs(mean_val - np.min(data[y][mask])))
            max_vals.append(np.abs(np.max(data[y][mask]) - mean_val))

        # print(means)

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle(f"Effect of '{fname}'", fontsize=FONTSIZE_TITLE)
        axs.errorbar(
            levels,
            means,
            yerr=np.array([min_vals, max_vals]),
            linewidth=3,
            elinewidth=3,
            # label="Min. and Max. values at errorbars",
        )
        # axs.legend()
        axs.set_xticks(levels)
        axs.set_xlabel(fname, fontsize=FONTSIZE_AXES)
        axs.set_ylabel(y, fontsize=FONTSIZE_AXES)
        axs.grid()
        # axs.set_ylim(np.min(data[y]), np.max(data[y]))
        fig.tight_layout()
        if save_path:
            fig.savefig(
                save_path + f"/{save_prefix}_{fname}_main_effect.png",
                dpi=300,
                bbox_inches="tight",
            )


def create_interaction_plots(
    data, y, figsize=FIGSIZE_FULL, save_path=None, save_prefix=None
):
    factors = [col for col in data.columns if col != y]

    factor_pairs = combinations(factors, 2)

    # Get levels
    levels = {f: [] for f in factors}
    for fi, fname in enumerate(factors):
        levels[fname].append(np.unique(data[factors].values))

    for f1, f2 in factor_pairs:
        levels_f1 = np.unique(data[f1])
        levels_f2 = np.unique(data[f2])

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        for l2 in levels_f2:
            means = []
            min_vals = []
            max_vals = []

            for l1 in levels_f1:
                mask = (data[f1] == l1) & (data[f2] == l2)
                mean_val = np.mean(data[y][mask])
                means.append(mean_val)
                min_vals.append(np.abs(mean_val - np.min(data[y][mask])))
                max_vals.append(np.abs(np.max(data[y][mask]) - mean_val))
            axs.plot(levels_f1, means, marker="o", label=f"{f2}={int(l2)}")
        axs.set_xticks(levels_f1)
        axs.set_xlabel(f1, fontsize=FONTSIZE_AXES)
        axs.set_ylabel(f"Mean {y}", fontsize=FONTSIZE_AXES)
        # axs.set_ylim(np.min(data[y]), np.max(data[y]))
        fig.suptitle(f"{f1} : {f2}", fontsize=FONTSIZE_TITLE)
        axs.legend()
        axs.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        if save_path:
            fig.savefig(
                save_path + f"/{save_prefix}_{f1}_{f2}_interaction_effect.png",
                dpi=300,
                bbox_inches="tight",
            )


# 'Filtration_rate ~ T + CoF + RPM + T:CoF + T:RPM'
def generate_anova_formula(response_name, factor_names, max_interactions):
    if max_interactions <= 0 or max_interactions > len(factor_names):
        raise ValueError(
            "'max_interactions' must be in range 0 < x <= len('factor_names')"
        )

    model_formula = ""
    model_formula += response_name + " ~ "

    combined_factors_at_interaction_num = []
    for interaction_num in range(1, max_interactions + 1):
        for combo in combinations(factor_names, interaction_num):
            combined_factors_at_interaction_num.append(":".join(combo))

    model_formula += " + ".join(combined_factors_at_interaction_num)
    return model_formula


def mapp_design_to_real_values(design, factors):
    mapped_design = {}
    for ci, k in enumerate(factors.keys()):
        mapped_design[k] = []
        for ri in range(len(design)):
            mapped_design[k].append(factors[k][int(design[ri][ci])])
    return mapped_design


def run_anova(mapped_experiments, factors, target, n_interactions, anova_typ):
    anova_model_formula = generate_anova_formula(
        target, factors.keys(), max_interactions=n_interactions
    )
    model = ols(anova_model_formula, data=mapped_experiments).fit()
    anova_results = sm.stats.anova_lm(model, typ=anova_typ)
    return model, anova_results


def calculate_effects(data, factors, response, max_interactions):
    effects = {}
    n = len(data)

    # Iterate through all main factors and their interactions
    for r in range(1, min(len(factors), max_interactions) + 1):
        for combination in combinations(factors, r):
            term = "*".join(combination)
            interaction_term = data[list(combination)].prod(
                axis=1
            )  # Multiply factor columns to get interaction
            effect = (interaction_term * data[response]).sum() / (
                n / 2
            )  # Calculate effect
            effects[term] = effect

    return pd.DataFrame({"Factor": effects.keys(), "Effect": effects.values()})
