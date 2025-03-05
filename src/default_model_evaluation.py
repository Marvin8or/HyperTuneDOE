import numpy as np
import pandas as pd
import doe_utils as du
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score

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

kf = KFold(n_splits=du.NSPLITS, shuffle=True, random_state=du.RSTATE)

model_results = {
    # "RandomForestRegressor": {"MSE": [], "R2": [], "Residuals": []},
    "DecisionTreeRegressor_best": {"MSE": [], "R2": [], "Residuals": []},
    "DecisionTreeRegressor_grid_search": {
        "MSE": [],
        "R2": [],
        "Residuals": [],
    },
    # "DecisionTreeRegressor_worst": {"MSE": [], "R2": [], "Residuals": []},
    # "DecisionTreeRegressor_overfit": {"MSE": [], "R2": [], "Residuals": []},
    "DecisionTreeRegressor_underfit": {"MSE": [], "R2": [], "Residuals": []},
}
# Fitting 3 folds for each of 24192 candidates, totalling 72576 fits
# Best Parameters: {'ccp_alpha': 0.02, 'max_depth': 10, 'max_features': None, 'max_leaf_nodes': 100, 'min_impurity_decrease': 0, 'min_samples_leaf': 5, 'min_samples_split': 5, 'min_weight_fraction_leaf': 0}
# Best MSE: 63.29146048163526
# Final Test MSE: 49.210223519210814

best_params = {
    "min_samples_leaf": 14,
    "min_weight_fraction_leaf": 0.0043,
    "max_features": 7,
}

grid_search_best_params = {
    "max_depth": 10,
    "min_samples_leaf": 5,
    "min_weight_fraction_leaf": 0,
    "max_features": None,
    "min_samples_split": 5,
    "max_leaf_nodes": 100,
    "min_impurity_decrease": 0,
    "ccp_alpha": 0.02,
}

underfit_params = dict(
    max_depth=1,
    min_samples_split=len(X_train),
    min_samples_leaf=len(X_train) // 2,
    min_weight_fraction_leaf=0.5,
    max_features=1,
    max_leaf_nodes=2,
    ccp_alpha=1.0,
)

for model, model_name in zip(
    [
        DecisionTreeRegressor(random_state=du.RSTATE, **best_params),
        DecisionTreeRegressor(
            random_state=du.RSTATE, **grid_search_best_params
        ),
        DecisionTreeRegressor(random_state=du.RSTATE, **underfit_params),
    ],
    model_results.keys(),
):
    print("==============")
    print(model_name, " results:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MSE: ", np.round(mean_squared_error(y_test, y_pred), 3))
    print("R2: ", np.round(r2_score(y_test, y_pred), 3) * 100, "%")
    print(
        "cv score: ",
        -np.round(
            cross_val_score(model, X, y, scoring="neg_mean_squared_error"), 3
        ),
    )
    print("==============")


for model, model_name in zip(
    [
        DecisionTreeRegressor(random_state=du.RSTATE, **best_params),
        DecisionTreeRegressor(
            random_state=du.RSTATE, **grid_search_best_params
        ),
        DecisionTreeRegressor(random_state=du.RSTATE, **underfit_params),
    ],
    model_results.keys(),
):
    # print("Evaluation of ", model)
    # print()
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # print(f"Evaluating on split {i+1}")
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

        # print(
        #     f"Validation MSE(fold n={i+1}): {np.round(model_results[model_name]['MSE'][-1], 3)}"
        # )
        # print(
        #     f"Validation R2(fold n={i+1}): {np.round(model_results[model_name]['R2'][-1]*100, 3)}\n"
        # )
for model_name in model_results.keys():
    model_results[model_name]["Residuals"] = np.hstack(
        model_results[model_name]["Residuals"]
    )
    print("Mean metrics for ", model_name)
    print()
    print(
        f"Mean MSE over all folds: {np.round(np.mean(model_results[model_name]['MSE']), 3)}"
    )
    print(
        f"Mean R2 over all folds: {np.round(np.mean(model_results[model_name]['R2'])*100, 3)}"
    )

# %% Bar plots over cvs
fig, axs = plt.subplots(2, 1, figsize=du.FIGSIZE_FULL)
bar_width = 0.2
x = np.array([1 - bar_width, 2, 3])
axs[0].bar(
    x - bar_width,
    model_results["DecisionTreeRegressor_best"]["MSE"],
    bar_width,
    color="greenyellow",
    label="Tuned DTR",
)
axs[0].bar(
    x,
    model_results["DecisionTreeRegressor_grid_search"]["MSE"],
    bar_width,
    color="green",
    label="GridSearch DTR",
)
axs[0].bar(
    x + bar_width,
    model_results["DecisionTreeRegressor_underfit"]["MSE"],
    bar_width,
    color="red",
    alpha=0.6,
    label="Underfitted DTR",
)

axs[1].bar(
    x - bar_width,
    model_results["DecisionTreeRegressor_best"]["R2"],
    bar_width,
    color="greenyellow",
    label="Tuned DTR",
)
axs[1].bar(
    x,
    model_results["DecisionTreeRegressor_grid_search"]["R2"],
    bar_width,
    color="green",
    label="GridSearch DTR",
)
axs[1].bar(
    x + bar_width,
    np.abs(model_results["DecisionTreeRegressor_underfit"]["R2"]) * 10,
    bar_width,
    color="red",
    alpha=0.6,
    label="Underfitted DTR",
)

axs[0].legend()
axs[1].legend()
axs[0].grid()
axs[1].grid()
axs[0].set_title("MSE over validation sets", fontsize=du.FONTSIZE_AXES)
axs[1].set_title("R² over validation sets", fontsize=du.FONTSIZE_AXES)
axs[0].set_xticks(x)
axs[0].set_xticklabels(["k=1", "k=2", "k=3"], fontsize=du.FONTSIZE_AXES - 2)
axs[1].set_xticks(x)
axs[1].set_xticklabels(["k=1", "k=2", "k=3"], fontsize=du.FONTSIZE_AXES - 2)
plt.tight_layout()

# %% Bar plot of evaluations
evals = [150, 72576]
fig, axs = plt.subplots(1, 1, figsize=du.FIGSIZE_FULL)
bar_width = 0.5
x = [1, 2]

axs.bar(x[0], evals[0], bar_width, color="green", label="Tuned DTR")
axs.bar(x[1], evals[1], bar_width, color="red", alpha=0.5, label="GS DTR")
# %% Plotting histograms of residuals


# fig, axs = plt.subplots(
#     1, 3, sharex=True, sharey=True, figsize=du.FIGSIZE_FULL
# )
# axs[0].set_title("RFR\nResiduals", fontsize=du.FONTSIZE_TITLE)
# axs[0].hist(
#     model_results["RandomForestRegressor"]["Residuals"],
#     # label=model_names[1],
#     color="skyblue",
#     bins=30,
#     # alpha=0.7,
#     edgecolor="black",
# )

# axs[1].set_title("Baseline DTR\nResiduals", fontsize=du.FONTSIZE_TITLE)
# axs[1].hist(
#     model_results["DecisionTreeRegressor_worst"]["Residuals"],
#     # label=model_names[0],
#     color="skyblue",
#     bins=30,
#     # alpha=0.7,
#     edgecolor="black",
# )

# axs[2].set_title("Tuned DTR\nResiduals", fontsize=du.FONTSIZE_TITLE)
# axs[2].hist(
#     model_results["DecisionTreeRegressor_best"]["Residuals"],
#     # label=model_names[0],
#     color="skyblue",
#     bins=30,
#     # alpha=0.7,
#     edgecolor="black",
# )
# for i in range(len(axs)):
#     axs[i].set_xlabel("Residuals", fontsize=du.FONTSIZE_AXES)
#     axs[i].grid()
# axs[0].set_ylabel("Frequency", fontsize=du.FONTSIZE_AXES)
# plt.tight_layout()


# %% Box plots of MSE and R2
model_names = [
    "Tuned DTR",
    "GridSearch DTR",
    "Underfitted DTR",
]
fig, axs = plt.subplots(1, 2, sharex=True, figsize=du.FIGSIZE_FULL)
mse_values = [model_results[mname]["MSE"] for mname in model_results.keys()]
colors = ["skyblue", "skyblue", "skyblue"]
axs[0].set_ylabel("Median MSE over validation sets", fontsize=du.FONTSIZE_AXES)
median_props = dict(color="black", linestyle="--", linewidth=1)

bplot = axs[0].boxplot(mse_values, patch_artist=True, medianprops=median_props)

axs[0].set_xticks(
    [1, 2, 3]
)  # Positions of the ticks (1 for Model A, 2 for Model B)
axs[0].set_xticklabels(
    model_names, rotation=30, ha="right", fontsize=du.FONTSIZE_AXES
)  # Custom labels and rotation
axs[0].grid()

for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)

r2_values = []

for mname in model_results.keys():
    if mname == "DecisionTreeRegressor_underfit":
        r2_values.append(np.abs(model_results[mname]["R2"]) * 10)
    else:
        r2_values.append(np.abs(model_results[mname]["R2"]))
axs[1].set_ylabel("Median R² over validation sets", fontsize=du.FONTSIZE_AXES)


bplot = axs[1].boxplot(r2_values, patch_artist=True, medianprops=median_props)

axs[1].set_xticks(
    [1, 2, 3]
)  # Positions of the ticks (1 for Model A, 2 for Model B)
model_names = [
    "Tuned DTR",
    "GridSearch DTR",
    "Underfitted DTR",
]

axs[1].set_xticklabels(
    model_names, rotation=30, ha="right", fontsize=du.FONTSIZE_AXES
)  # Custom labels and rotation
axs[1].grid()
for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)
plt.tight_layout()
