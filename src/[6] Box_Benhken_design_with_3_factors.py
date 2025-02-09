# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:52:46 2025

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
# factors_with_center_points = {
#     "max_depth": {-1: 2, 0: 20, 1: 30},
#     "min_weight_fraction_leaf": {-1: 0, 0: 0.1, 1: 0.5},
#     "max_features": {-1: 1, 0: 4, 1: 8},
# }
factors_with_center_points = {
    "max_depth": {-1: 20, 0: 25, 1: 30},
    "min_weight_fraction_leaf": {-1: 0.2, 0: 0.25, 1: 0.3},
    "max_features": {-1: 3, 0: 4, 1: 5},
}
design = doe3.bbdesign(3)

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
mapped_results = pd.concat([mapped_design_df, results], axis=1)

# %% Fit Linear model to quadratic features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

X = results[["max_depth", "min_weight_fraction_leaf", "max_features"]]
y = results["MSE"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Fit the quadratic model
model = LinearRegression()

model.fit(X_poly, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

r2 = model.score(X_poly, y)
print(f"R²: {r2:.4f}")

# Calculate Adjusted R²
n = X.shape[0]  # Number of observations
p = X_poly.shape[1]  # Number of predictors
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Adjusted R²: {adj_r2:.4f}")

# Add column names for clarity
X_poly_df = pd.DataFrame(
    X_poly,
    columns=[
        "max_depth",
        "min_weight_fraction_leaf",
        "max_features",
        "max_depth:min_weight_fraction_leaf",
        "max_depth:max_features",
        "min_weight_fraction_leaf:max_features",
        "I(max_depth**2)",
        "I(min_weight_fraction_leaf**2)",
        "I(max_features**2)",
    ],
)
formula = "MSE ~ " + " + ".join(X_poly_df.columns)

X_poly_df["MSE"] = results["MSE"]
print("Quadratic Formula:", formula)

model = ols(formula, data=X_poly_df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
# No significant interactions and quadratic terms!!

# %% Try anova with log
# X_poly_transformed = X_poly_df.copy()
# X_poly_transformed["MSE"] = np.log(X_poly_transformed["MSE"])
# model = ols(formula, data=X_poly_transformed).fit()
# anova_results_log = sm.stats.anova_lm(model, typ=2)

# %% Try anova with sqrt
# X_poly_transformed_sqrt = X_poly_df.copy()
# X_poly_transformed_sqrt["MSE"] = np.sqrt(X_poly_transformed_sqrt["MSE"])
# model = ols(formula, data=X_poly_transformed_sqrt).fit()
# anova_results_sqrt = sm.stats.anova_lm(model, typ=2)

# %% Try anova with Box-Cox transformation
# from scipy.stats import boxcox

# X_poly_transformed_boxcox = X_poly_df.copy()

# MSE_transformed, best_lambda = boxcox(X_poly_df["MSE"])
# print(f"Optimal lambda: {best_lambda}")

# X_poly_transformed_boxcox["MSE"] = MSE_transformed
# model = ols(formula, data=X_poly_transformed_boxcox).fit()
# anova_results_boxcox = sm.stats.anova_lm(model, typ=2)

# %% Ridge analysis
from scipy.optimize import minimize

parameters = model.params


def quadratic_model(x):
    max_depth, min_weight_fraction_leaf, max_features = x
    return (
        parameters["Intercept"]
        + parameters["max_depth"] * max_depth
        + parameters["min_weight_fraction_leaf"] * min_weight_fraction_leaf
        + parameters["max_features"] * max_features
        + parameters["I(max_depth ** 2)"] * max_depth**2
        + parameters["I(min_weight_fraction_leaf ** 2)"]
        * min_weight_fraction_leaf**2
        + parameters["I(max_features ** 2)"] * max_features**2
        + parameters["max_depth:min_weight_fraction_leaf"]
        * max_depth
        * min_weight_fraction_leaf
        + parameters["max_depth:max_features"] * max_depth * max_features
        + parameters["min_weight_fraction_leaf:max_features"]
        * min_weight_fraction_leaf
        * max_features
    )


def constraint(x, r):
    max_depth, min_weight_fraction_leaf, max_features = x
    return (
        max_depth**2
        + min_weight_fraction_leaf**2
        + max_features**2
        - r**2
    )


def gradient(x):
    max_depth, min_weight_fraction_leaf, max_features = x
    grad_max_depth = (
        parameters["max_depth"]
        + 2 * parameters["I(max_depth ** 2)"] * max_depth
        + parameters["max_depth:min_weight_fraction_leaf"]
        * min_weight_fraction_leaf
        + parameters["max_depth:max_features"] * max_features
    )
    grad_min_weight_fraction_leaf = (
        parameters["min_weight_fraction_leaf"]
        + 2
        * parameters["I(min_weight_fraction_leaf ** 2)"]
        * min_weight_fraction_leaf
        + parameters["max_depth:min_weight_fraction_leaf"] * max_depth
        + parameters["min_weight_fraction_leaf:max_features"] * max_features
    )
    grad_max_features = (
        parameters["max_features"]
        + 2 * parameters["I(max_features ** 2)"] * max_features
        + parameters["max_depth:max_features"] * max_depth
        + parameters["min_weight_fraction_leaf:max_features"]
        * min_weight_fraction_leaf
    )
    return np.array(
        [grad_max_depth, grad_min_weight_fraction_leaf, grad_max_features]
    )


def ridge_analysis_descent(r, p0, step_size=1, num_steps=3):
    points = [p0]

    for _ in range(num_steps):
        grad = gradient(points[-1])  # Compute gradient
        step = -step_size * (
            grad / np.linalg.norm(grad)
        )  # Move in steepest descent direction

        # Ensure we stay within the ridge constraint
        new_point = points[-1] + step
        points.append(new_point)

    return np.array(points)


p0_1 = np.array([25.0, 0.22, 4.0])

r = 21
steps = ridge_analysis_descent(r, p0_1, num_steps=1000)
