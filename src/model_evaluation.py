import numpy as np
import pandas as pd
import doe_utils as du
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold

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
X, y = (
    df[[c for c in df.columns if c != "concrete_compressive_strength"]],
    df["concrete_compressive_strength"],
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=du.TSIZE, shuffle=True, random_state=du.RSTATE
)

kf = KFold(n_splits=du.NSPLITS, shuffle=True)

model_results = {
    "RandomForestRegressor": {"MSE": [], "R2": [], "Residuals": []},
    "DecisionTreeRegressor": {"MSE": [], "R2": [], "Residuals": []},
}


for model, model_name in zip(
    [
        RandomForestRegressor(random_state=du.RSTATE),
        DecisionTreeRegressor(
            random_state=du.RSTATE,
            # **dict(
            #     max_depth=10,
            #     min_weight_fraction_leaf=0.01,
            #     min_samples_leaf=1,
            #     min_samples_split=5,
            #     max_features=8,
            #     ccp_alpha=0.001,
            # ),
        ),
    ],
    ["RandomForestRegressor", "DecisionTreeRegressor"],
):
    print("Evaluation of ", model)
    print()
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Evaluating on split {i+1}")
        Xtrain, Xval = X_train.values[train_idx], X_train.values[val_idx]

        ytrain, yval = y_train.values[train_idx], y_train.values[val_idx]

        fitted_model = model.fit(Xtrain, ytrain.ravel())
        yvalpred = fitted_model.predict(Xval)

        model_results[model_name]["MSE"].append(
            mean_squared_error(yval, yvalpred)
        )
        model_results[model_name]["R2"].append(r2_score(yval, yvalpred))
        model_results[model_name]["Residuals"].append(yval - yvalpred)

        if False:
            plt.figure()
            plt.title(f"Fold {i+1}")
            plt.plot(yvalpred, "bo", label="Predicted values")
            plt.plot(yval, "ro", label="Ground truth")
            plt.grid()
            plt.legend()

        print(
            f"Validation MSE(fold n={i+1}): {np.round(model_results[model_name]['MSE'][-1], 3)}"
        )
        print(
            f"Validation R2(fold n={i+1}): {np.round(model_results[model_name]['R2'][-1]*100, 3)}\n"
        )
for model, model_name in zip(
    [
        RandomForestRegressor(random_state=du.RSTATE),
        DecisionTreeRegressor(random_state=du.RSTATE),
    ],
    ["RandomForestRegressor", "DecisionTreeRegressor"],
):
    model_results[model_name]["Residuals"] = np.hstack(
        model_results[model_name]["Residuals"]
    )
    print("Mean metrics for ", model)
    print()
    print(
        f"Mean MSE over all folds: {np.round(np.mean(model_results[model_name]['MSE']), 3)}"
    )
    print(
        f"Min MSE over all folds: {np.round(np.min(model_results[model_name]['MSE']), 3)}"
    )
    print(
        f"Mean R2 over all folds: {np.round(np.mean(model_results[model_name]['R2'])*100, 3)}"
    )
    print(
        f"Max R2 over all folds: {np.round(np.max(model_results[model_name]['R2'])*100, 3)}"
    )

# %% Plotting histograms of residuals
fontsize_axes = 12
fontsize_title = 15
model_names = ["Random Forest", "Decision Tree"]
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5.625))
axs[0].set_title("DecisionTreeRegressor\nResiduals", fontsize=fontsize_title)
axs[0].hist(
    model_results["DecisionTreeRegressor"]["Residuals"],
    # label=model_names[0],
    color="skyblue",
    bins=30,
    # alpha=0.7,
    edgecolor="black",
)
axs[1].set_title("RandomForestRegressor\nResiduals", fontsize=fontsize_title)
axs[1].hist(
    model_results["RandomForestRegressor"]["Residuals"],
    # label=model_names[1],
    color="skyblue",
    bins=30,
    # alpha=0.7,
    edgecolor="black",
)
axs[0].set_xlabel("Residuals", fontsize=fontsize_axes)
axs[1].set_xlabel("Residuals", fontsize=fontsize_axes)
axs[0].grid()
axs[1].grid()
axs[0].set_ylabel("Frequency", fontsize=fontsize_axes)
plt.tight_layout()


# %% Box plots of MSE and R2

fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 5.625))
mse_values = [model_results[mname]["MSE"] for mname in model_results.keys()]
colors = ["skyblue", "skyblue"]
axs[0].set_ylabel("Mean Squared Error over folds", fontsize=fontsize_axes)
median_props = dict(color="black", linestyle="--", linewidth=1)

bplot = axs[0].boxplot(mse_values, patch_artist=True, medianprops=median_props)

axs[0].set_xticks(
    [1, 2]
)  # Positions of the ticks (1 for Model A, 2 for Model B)
axs[0].set_xticklabels(
    model_names, rotation=30, ha="right", fontsize=fontsize_axes
)  # Custom labels and rotation
axs[0].set_ylim(10, 70)
axs[0].grid()

for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)

r2_values = [model_results[mname]["R2"] for mname in model_results.keys()]
axs[1].set_ylabel("R² Error over folds", fontsize=fontsize_axes)


bplot = axs[1].boxplot(r2_values, patch_artist=True, medianprops=median_props)

axs[1].set_xticks(
    [1, 2]
)  # Positions of the ticks (1 for Model A, 2 for Model B)
axs[1].set_xticklabels(
    model_names, rotation=30, ha="right", fontsize=fontsize_axes
)  # Custom labels and rotation
axs[1].grid()
axs[1].set_ylim(0.75, 1)
for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)
plt.tight_layout()
