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
    "max_depth": {-1: 2, 1: 20},
    "min_samples_split": {-1: 2, 1: 50},
    "min_samples_leaf": {-1: 1, 1: 20},
    "min_weight_fraction_leaf": {-1: 0, 1: 0.2},
    "max_features": {-1: 1, 1: 8},
    "max_leaf_nodes": {-1: 100, 1: 1000},
    "ccp_alpha": {-1: 0, 1: 0.2},
}

design = doe3.ff2n(len(factors.keys()))
print("\n".join(doe3.fracfact_aliasing(design)[0]))

design_df = pd.DataFrame(columns=list(factors.keys()), data=design)

mapped_design = du.mapp_design_to_real_values(design, factors)
mapped_design_df = pd.DataFrame(mapped_design)
mapped_design_df.max_leaf_nodes = mapped_design_df.max_leaf_nodes.replace(
    np.nan, None
)
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

if False:
    du.create_main_effects_plots(results, du.TARGET, [du.TARGET])
    du.create_interaction_plots(results, du.TARGET, [du.TARGET])

anova_model, anova_results = du.run_anova(
    results, factors, du.TARGET, n_interactions=3, anova_typ=2
)

# %% Analysis of effects and model residuals
from itertools import combinations
import statsmodels.api as sm


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


df_effects = calculate_effects(results, factors, "MSE", 3)
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
sm.qqplot(df_effects["Effect"], line="q", ax=axs)
probplot = sm.ProbPlot(df_effects["Effect"])  # Compute QQ plot values
outlier_mask = np.abs(
    df_effects["Effect"] - np.median(df_effects["Effect"])
) > 1.1 * np.std(df_effects["Effect"])
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

plt.show()


# Residuals vs Predicted Responses
predicted_responses = anova_model.predict()
residuals = anova_model.resid


fig, ax = plt.subplots()
ax.scatter(predicted_responses, residuals)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted Responses")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Predicted Responses")
plt.show()

# --- QQ plot of residuals ---
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
sm.qqplot(residuals, line="q", ax=axs)

# --- Box plot of residuals ---
plt.figure(figsize=(8, 6))  # Adjust figure size if needed
plt.boxplot(
    residuals, vert=False, patch_artist=True
)  # `vert=False` for horizontal boxplot
plt.xlabel("Residuals")
plt.title("Box Plot of Residuals")
plt.show()

# --- Histogram of residuals ---
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins="auto", edgecolor="black")  # Adjust `bins` if needed
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# %% Distilled full factorial and analysis of effects and residuals of model

factors_with_center_points = {
    "max_depth": {-1: 2, 0: 10, 1: 20},
    "min_weight_fraction_leaf": {-1: 0, 0: 0.1, 1: 0.2},
    "max_features": {-1: 1, 0: 4, 1: 8},
}

design = doe3.ff2n(len(factors_with_center_points.keys()))
centerpoints = np.array(
    [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
    ]
)
design = np.vstack([design, centerpoints])

np.random.shuffle(design)
design_df = pd.DataFrame(
    columns=list(factors_with_center_points.keys()), data=design
)

mapped_design = du.mapp_design_to_real_values(
    design, factors_with_center_points
)
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

# Anova
anova_model, anova_results = du.run_anova(
    results,
    factors_with_center_points,
    du.TARGET,
    n_interactions=3,
    anova_typ=2,
)


df_effects = calculate_effects(results, factors_with_center_points, "MSE", 2)
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
sm.qqplot(df_effects["Effect"], line="q", ax=axs)
probplot = sm.ProbPlot(df_effects["Effect"])  # Compute QQ plot values
outlier_mask = np.abs(
    df_effects["Effect"] - np.median(df_effects["Effect"])
) > 1.5 * np.std(df_effects["Effect"])
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

plt.show()


# Residuals vs Predicted Responses
predicted_responses = anova_model.predict()
residuals = anova_model.resid


fig, ax = plt.subplots()
ax.scatter(predicted_responses, residuals)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted Responses")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Predicted Responses")
plt.show()

# --- QQ plot of residuals ---
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
sm.qqplot(residuals, line="q", ax=axs)

# --- Box plot of residuals ---
plt.figure(figsize=(8, 6))  # Adjust figure size if needed
plt.boxplot(
    residuals, vert=False, patch_artist=True
)  # `vert=False` for horizontal boxplot
plt.xlabel("Residuals")
plt.title("Box Plot of Residuals")
plt.show()

# --- Histogram of residuals ---
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins="auto", edgecolor="black")  # Adjust `bins` if needed
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

if True:
    du.create_main_effects_plots(results, du.TARGET, [du.TARGET])
    du.create_interaction_plots(results, du.TARGET, [du.TARGET])
