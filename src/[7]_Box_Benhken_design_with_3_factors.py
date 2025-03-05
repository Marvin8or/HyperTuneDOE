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
# max_depth was added to explor really low settings
# =============================================================================

factors = {
    "min_samples_leaf": {-1: 18, 0: 28, 1: 38},
    "min_weight_fraction_leaf": {-1: 0.005, 0: 0.025, 1: 0.03},
    "max_features": {-1: 5, 0: 6, 1: 7},
}

design = doe3.bbdesign(3, center=1)

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
    # fixed_params=fixed_params,
    n_reps=5,
)
results = pd.concat([design_df, results["MSE"]], axis=1)
mapped_results = pd.concat([mapped_design_df, results["MSE"]], axis=1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = results[["min_samples_leaf", "min_weight_fraction_leaf", "max_features"]]
y = results["MSE"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X.values)
feature_names = poly.get_feature_names_out(
    ["min_samples_leaf", "min_weight_fraction_leaf", "max_features"]
)
X_poly_df = pd.DataFrame(
    X_poly,
    columns=feature_names,
).replace({-0: 0})

formula = "MSE ~ " + " + ".join(X_poly_df.columns)

print("Quadratic Formula:", formula)

model = LinearRegression().fit(X_poly_df.values, y.values)
best_combination = X_poly_df.iloc[np.argmin(results["MSE"])]


def quadratic_model(X, coefficients, intercept):
    predictions = (
        intercept
        + coefficients[0] * X[0]
        + coefficients[1] * X[1]
        + coefficients[2] * X[2]
        + coefficients[3] * X[3]
        + coefficients[4] * X[4]
        + coefficients[5] * X[5]
        + coefficients[6] * X[6]
        + coefficients[7] * X[7]
        + coefficients[8] * X[8]
    )

    return predictions


print(quadratic_model(best_combination.values, model.coef_, model.intercept_))
print(model.predict(best_combination.values.reshape(1, -1)))


def gradients(X, coefficients):
    x1, x2, x3 = X[0], X[1], X[2]
    grad_x1 = (
        coefficients[0]
        + coefficients[3] * x2
        + coefficients[4] * x3
        + 2 * coefficients[6] * x1
    )

    grad_x2 = (
        coefficients[1]
        + coefficients[3] * x1
        + coefficients[5] * x3
        + 2 * coefficients[7] * x2
    )

    grad_x3 = (
        coefficients[2]
        + coefficients[4] * x1
        + coefficients[5] * x2
        + 2 * coefficients[8] * x3
    )

    return np.array([grad_x1, grad_x2, grad_x3])


def enforce_constraint(x, r):
    return (x / np.linalg.norm(x)) * r


def normalize(x):
    return x / np.linalg.norm(x)


def interpolate_encoded_values(x, a, b):
    return a * x + b


def min_samples_leaf_f(x):
    return interpolate_encoded_values(x, 10, 28)


def min_weight_fraction_leaf_f(x):
    return interpolate_encoded_values(x, 0.025, 0.005)


def max_features_f(x):
    return interpolate_encoded_values(x, 1, 6)


def iterative_gradient_descent(x, eta, model, model_params, n_iter, r):
    # points = pd.DataFrame([x.values], columns=x.index)
    points = [x.values]
    results = [model.predict(x.values.reshape(1, -1))]
    for ni in range(n_iter):
        current_point = points[-1]

        grad_x1 = gradients(current_point, model.coef_)[0]
        grad_x2 = gradients(current_point, model.coef_)[1]
        grad_x3 = gradients(current_point, model.coef_)[2]

        new_x1 = current_point[0] - eta * grad_x1
        new_x2 = current_point[1] - eta * grad_x2
        new_x3 = current_point[2] - eta * grad_x3

        new_point = np.array([new_x1, new_x2, new_x3])

        new_point = poly.transform(new_point.reshape(1, -1))
        if np.linalg.norm(new_point[0] - current_point) < r:
            break

        points.append(new_point[0])
        results.append(model.predict(new_point))

    col_names = [k for k in x.index]
    result_df = pd.DataFrame(columns=col_names)
    result_df[col_names] = np.vstack(points)
    result_df["MSE"] = np.hstack(results)

    results_interp_df = pd.DataFrame()

    results_interp_df["min_samples_leaf"] = [
        min_samples_leaf_f(x) for x in result_df["min_samples_leaf"].values
    ]
    results_interp_df["min_weight_fraction_leaf"] = [
        min_weight_fraction_leaf_f(x)
        for x in result_df["min_weight_fraction_leaf"].values
    ]
    results_interp_df["max_features"] = [
        max_features_f(x) for x in result_df["max_features"].values
    ]
    results_interp_df["MSE"] = result_df["MSE"]
    return result_df, results_interp_df


RESLTS, RESULTS_INTERP = iterative_gradient_descent(
    best_combination, 0.005, model, model.coef_, 5, 0.1
)
print(RESULTS_INTERP["MSE"], RESULTS_INTERP["min_samples_leaf"])
# %% Ridge analysis

min_samples_leaf_vals = RESLTS["min_samples_leaf"]
min_weight_fraction_leaf_vals = RESLTS["min_weight_fraction_leaf"]
max_features_vals = RESLTS["max_features"]

X1, X2 = np.meshgrid(min_samples_leaf_vals, min_weight_fraction_leaf_vals)

## Compute target values
Z = np.zeros(X1.shape)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = model.predict(
            poly.transform(np.array([[X1[i, j], X2[i, j], 1]]))
        )[0]

# Create the contour plot (ridge-like visualization)
plt.figure(figsize=du.FIGSIZE_FULL)
contour = plt.contourf(X1, X2, Z, levels=30, cmap="viridis")  # Smooth ridges
cbar = plt.colorbar(contour, label="MSE")

# Optional: Add contour lines
contour_lines = plt.contour(
    X1, X2, Z, levels=10, colors="black", linewidths=0.5
)
plt.clabel(contour_lines, inline=True, fontsize=10, fmt="%.2f")

# Labels and title
plt.xlabel("min_samples_leaf", fontsize=du.FONTSIZE_AXES)
plt.ylabel("min_weight_fraction_leaf", fontsize=du.FONTSIZE_AXES)
# plt.title(
#     "Ridge Plot of MSE values of quadratic model", fontsize=du.FONTSIZE_TITLE
# )

plt.show()

# %% Evaluate Decision tree with bes fit
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X, y = (
    df[[c for c in df.columns if c != "concrete_compressive_strength"]],
    df["concrete_compressive_strength"],
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=du.TSIZE, shuffle=True, random_state=du.RSTATE
)

best_params = {
    "min_samples_leaf": 14,
    "min_weight_fraction_leaf": 0.0043,
    "max_features": 7,
}

model = DecisionTreeRegressor(**best_params, random_state=du.RSTATE)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2: ", r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print(
    "cross val R2: ",
    du.evaluate_model(X, y, "R2", model_hyperparams=best_params),
)
print(
    "cross val MSE: ",
    du.evaluate_model(X, y, "MSE", model_hyperparams=best_params),
)

# Default values
# print(
#     "cross val MSE: ",
#     du.evaluate_model(X, y, "MSE", model_hyperparams=None),
# )
# %% Save
results_path = "C:/Projects/MZIR/experiments_results"
results.to_excel(results_path + "/" + "[7] BB_results_R2.xlsx")
RESULTS_INTERP.to_excel(results_path + "/" + "[7] BB_results_R2_GD.xlsx")
