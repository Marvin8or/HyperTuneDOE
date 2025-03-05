# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:23:17 2025

@author: gabri
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

# TODO rename variables
# %% Plotting
import seaborn as sns

if False:
    for column in df.columns:
        fig, ax = plt.subplots(1, 1)
        ax.plot(df[column], "bo", label=column)
        ax.grid()
        ax.legend()

    # Correlation between columns
    sns.pairplot(df)

# No significant correlation between columns

# %% Check for nan values

print(df.isna().any())
# No nan values

# %% Train default model
from sklearn.ensemble import RandomForestRegressor, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import doe_utils as du

# from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = (
    df[[c for c in df.columns if c != "concrete_compressive_strength"]],
    df["concrete_compressive_strength"],
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=du.TSIZE, shuffle=True, random_state=du.RSTATE
)

kf = KFold(n_splits=du.NSPLITS, shuffle=True)
kf_maes = []

# model = RandomForestRegressor(random_state=du.RSTATE)
model = DecisionTreeRegressor(random_state=du.RSTATE)

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Evaluating on split {i+1}")
    Xtrain, Xval = X_train.values[train_idx], X_train.values[val_idx]
    # print(f"Number of X samples in training data: {len(Xtrain)}")
    # print(f"Number of X samples in validation data: {len(Xval)}")

    ytrain, yval = y_train.values[train_idx], y_train.values[val_idx]
    # print(f"Number of y samples in training data: {len(ytrain)}")
    # print(f"Number of y samples in validation data: {len(yval)}")

    fitted_model = model.fit(Xtrain, ytrain.ravel())
    yvalpred = fitted_model.predict(Xval)
    MSE = mean_squared_error(yval, yvalpred)
    kf_maes.append(MSE)

    plt.figure()
    plt.title(f"Fold {i+1}")
    plt.plot(yvalpred, "bo", label="Predicted values")
    plt.plot(yval, "ro", label="Ground truth")
    plt.grid()
    plt.legend()

    print(f"Validation MSE(fold n={i+1}): {MSE}")

print(f"Mean MSE: {np.round(np.mean(kf5_maes), 3)}")
rfr_model = make_pipeline(
    StandardScaler(), RandomForestRegressor(random_state=12)
)
# print(f"Number of X samples in training data: {len(X_train)}")
# print(f"Number of X samples in test data: {len(X_test)}")
fitted_model = rfr_model.fit(X_train.values, y_train.values.ravel())
model_prediction = fitted_model.predict(X_test.values)
print(
    f"Model MSE on entire dataset: {mean_squared_error(y_test, model_prediction)}"
)

if True:
    plt.figure()
    plt.title("Entire dataset")
    plt.plot(model_prediction, "bo", label="Predicted values")
    plt.plot(y_test.values, "ro", label="Ground truth")
    plt.grid()
    plt.legend()
