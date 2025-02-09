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
NSPLITS = 10
TARGET = "MSE"


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

    repetition_cols = [
        f"{target}_{repetition_idx+1}"
        for repetition_idx in range(n_repetitions)
    ]
    results_df[target] = np.mean(results_df[repetition_cols], axis=1)
    results_df = results_df.drop(columns=repetition_cols)
    return results_df


def create_main_effects_plots(data, y, repetition_results: list):
    # Get all column names except for the response variable
    factors = [col for col in data.columns if col not in repetition_results]

    for fi, fname in enumerate(factors):
        means = []
        levels = [-1, 1]
        min_vals = []
        max_vals = []
        for l in levels:
            mask = data[fname] == l
            mean_val = np.mean(data[y][mask])
            means.append(mean_val)
            min_vals.append(np.abs(mean_val - np.min(data[y][mask])))
            max_vals.append(np.abs(np.max(data[y][mask]) - mean_val))

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle(f"Effect of '{fname}'")
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
        axs.set_xlabel(fname)
        axs.set_ylabel(y)
        axs.grid()
        # axs.set_ylim(np.min(data[y]), np.max(data[y]))

    plt.tight_layout()


def create_interaction_plots(data, y, repetition_results):
    factors = [col for col in data.columns if col not in repetition_results]

    factor_pairs = combinations(factors, 2)

    for f1, f2 in factor_pairs:
        levels_f1 = np.unique(data[f1])
        levels_f2 = np.unique(data[f2])

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
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
            # axs.errorbar(
            #     levels_f1,
            #     means,
            #     yerr=np.array([min_vals, max_vals]),
            #     linewidth=3,
            #     elinewidth=3,
            #     marker="o",
            #     label=f"{f2}={int(l2)}",
            #     alpha=0.5,
            #     capsize=7,
            #     capthick=5
            #     # label="Min. and Max. values at errorbars",
            # )
            axs.plot(levels_f1, means, marker="o", label=f"{f2}={int(l2)}")
        axs.set_xticks(levels_f1)
        axs.set_xlabel(f1)
        axs.set_ylabel(f"Mean {y}")
        axs.set_ylim(np.min(data[y]), np.max(data[y]))
        fig.suptitle(f"{f1} : {f2}")
        axs.legend()
        axs.grid(True, linestyle="--", alpha=0.6)


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
