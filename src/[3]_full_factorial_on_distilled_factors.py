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
# min_samples_leaf, min_weight_fraction_leaf, max_features
# =============================================================================
factors = {
    "min_samples_leaf": {0: 10, 1: 50, 2: 100},
    "min_weight_fraction_leaf": {0: 0.1, 1: 0.25, 2: 0.4},
    "max_features": {0: 2, 1: 4, 2: 6},
}

design = doe3.fullfact([len(factors.keys())] * 3)
print("\n".join(doe3.fracfact_aliasing(design)[0]))

design_df = pd.DataFrame(columns=list(factors.keys()), data=design)

mapped_design = du.mapp_design_to_real_values(design, factors)
mapped_design_df = pd.DataFrame(mapped_design)
# mapped_design_df[
#     ["max_depth", "max_features", "max_leaf_nodes"]
# ] = mapped_design_df[["max_depth", "max_features", "max_leaf_nodes"]].replace(
#     {np.nan: None}
# )
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
    # fixed_params=fixed_params,
    n_reps=5,
)
results = pd.concat([design_df, results], axis=1)
# mapped_results = pd.concat([mapped_design_df, results], axis=1)


# %% ANOVA on full factorial with centerpoints
import statsmodels.api as sm

anova_model, anova_results = du.run_anova(
    results, factors, du.TARGET, n_interactions=2, anova_typ=2
)

anova_results = du.round_dataframe(anova_results)
anova_results_all = anova_results.copy()
anova_results = anova_results[anova_results["PR(>F)"] < 0.05]
best_combination = results.iloc[np.argmin(results["MSE"].values)]
worst_combination = results.iloc[np.argmax(results["MSE"].values)]
df_effects = du.calculate_effects(results, factors, "MSE", 2)
print(anova_model.summary())
if False:
    fig, axs = plt.subplots(1, 1, figsize=du.FIGSIZE_FULL)
    sm.qqplot(df_effects["Effect"], line="q", ax=axs)
    probplot = sm.ProbPlot(df_effects["Effect"])  # Compute QQ plot values
    outlier_mask = np.abs(
        df_effects["Effect"] - np.median(df_effects["Effect"])
    ) > 0.95 * np.std(df_effects["Effect"])
    outliers = df_effects.loc[outlier_mask]

    # Annotate outliers
    for i, row in outliers.iterrows():
        x, y = probplot.theoretical_quantiles[i], row["Effect"]
        axs.annotate(
            row["Factor"],
            xy=(x, y),
            textcoords="offset points",
            xytext=(1, -1),
            ha="left",
        )
    plt.xlabel(
        "Theoretical Quantiles", fontsize=du.FONTSIZE_AXES
    )  # X-axis label
    plt.ylabel("Sample Quantiles", fontsize=du.FONTSIZE_AXES)  # Y-axis label
    plt.title("QQ Plot of effects", fontsize=du.FONTSIZE_TITLE)
    plt.grid()
    plt.show()

    # Residuals vs Predicted Responses
    predicted_responses = anova_model.predict()
    residuals = anova_model.resid

    fig, ax = plt.subplots(figsize=du.FIGSIZE_FULL)
    ax.scatter(predicted_responses, residuals)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Responses", fontsize=du.FONTSIZE_AXES)
    ax.set_ylabel("Residuals", fontsize=du.FONTSIZE_AXES)
    ax.set_title(
        "Residuals vs Predicted Responses", fontsize=du.FONTSIZE_TITLE
    )
    plt.grid()
    plt.show()

    # --- QQ plot of residuals ---
    fig, axs = plt.subplots(1, 1, figsize=du.FIGSIZE_FULL)
    sm.qqplot(residuals, line="q", ax=axs)
    plt.xlabel(
        "Theoretical Quantiles", fontsize=du.FONTSIZE_AXES
    )  # X-axis label
    plt.ylabel("Sample Quantiles", fontsize=du.FONTSIZE_AXES)  # Y-axis label
    plt.title("QQ Plot of residuals", fontsize=du.FONTSIZE_TITLE)
    plt.grid()
    plt.show()

    # --- Box plot of residuals ---
    median_props = dict(color="black", linestyle="--", linewidth=2)
    plt.figure(figsize=du.FIGSIZE_FULL)  # Adjust figure size if needed
    bplot = plt.boxplot(
        residuals, vert=False, patch_artist=True, medianprops=median_props
    )  # `vert=False` for horizontal boxplot
    for patch, color in zip(bplot["boxes"], ["skyblue"]):
        patch.set_facecolor(color)

    plt.xlabel("Residuals", fontsize=du.FONTSIZE_AXES)
    plt.title("Box Plot of Residuals", fontsize=du.FONTSIZE_TITLE)
    plt.grid()
    plt.show()

    # --- Histogram of residuals ---
    plt.figure(figsize=du.FIGSIZE_FULL)
    plt.hist(
        residuals, bins="auto", edgecolor="black", color="skyblue"
    )  # Adjust `bins` if needed
    plt.xlabel("Residuals", fontsize=du.FONTSIZE_AXES)
    plt.ylabel("Frequency", fontsize=du.FONTSIZE_AXES)
    plt.title("Histogram of Residuals", fontsize=du.FONTSIZE_TITLE)
    plt.show()

if True:
    du.create_main_effects_plots(
        results,
        du.TARGET,
        figsize=du.FIGSIZE_HALF,
        save_prefix="[3]",
        # save_path="C:/Users/gabri/OneDrive/Desktop/Dokumenti doktorat/MZIR/Seminarski rad/Images",
    )
    du.create_interaction_plots(
        results,
        du.TARGET,
        figsize=du.FIGSIZE_HALF,
        save_prefix="[3]",
        # save_path="C:/Users/gabri/OneDrive/Desktop/Dokumenti doktorat/MZIR/Seminarski rad/Images",
    )


# %% Save results
if False:
    results_path = "C:/Projects/MZIR/experiments_results"
    results.to_excel(
        results_path + "/" + "[3] FullFactDesign_screening_results.xlsx"
    )
    anova_results_all.to_excel(
        results_path + "/" + "[3] FullFactDesign_ANOVA_results.xlsx"
    )
