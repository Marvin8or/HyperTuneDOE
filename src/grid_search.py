import numpy as np
import pandas as pd
import doe_utils as du
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error


file_path = "C:/Projects/MZIR/datasets/Concrete_Data.xls"

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

X, y = (
    df[[c for c in df.columns if c != "concrete_compressive_strength"]],
    df["concrete_compressive_strength"],
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=du.TSIZE, shuffle=True, random_state=du.RSTATE
)

param_grid = {
    "max_depth": [5, 10, 50, 100, 150, 200, None],
    "min_samples_split": [5, 15, 25, 35],
    "min_samples_leaf": [5, 15, 25, 35],
    "min_weight_fraction_leaf": [0, 0.001, 0.2],
    "max_features": [None, "sqrt", "log2"],
    "max_leaf_nodes": [2, 10, 100, None],
    "ccp_alpha": [0, 0.001, 0.02],
    # "min_impurity_decrease": [0, 0.1],
}

tree = DecisionTreeRegressor(random_state=du.RSTATE)

grid_search = GridSearchCV(
    tree,
    param_grid,
    cv=du.NSPLITS,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best MSE:", -grid_search.best_score_)

best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred)

print("Final Test MSE:", final_mse)
