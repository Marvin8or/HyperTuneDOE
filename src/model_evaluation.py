# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:12:32 2025

@author: gabri
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


file_path = "C:/Projects/MZIR/datasets/Concrete_Data.xls"

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)
new_columns_names = {"Cement (component 1)(kg in a m^3 mixture)":"cement", #kg in a m^3 mixture
                     "Blast Furnace Slag (component 2)(kg in a m^3 mixture)":"blast_furnace_slag", #kg in a m^3 mixture
                     "Fly Ash (component 3)(kg in a m^3 mixture)":"fly_ash", #kg in a m^3 mixture
                     "Water  (component 4)(kg in a m^3 mixture)":"water", #kg in a m^3 mixture
                     "Superplasticizer (component 5)(kg in a m^3 mixture)":"superplasticizer", #kg in a m^3 mixture
                     "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":"coarse_aggregate", #kg in a m^3 mixture
                     "Fine Aggregate (component 7)(kg in a m^3 mixture)":"fine_aggregate", #kg in a m^3 mixture
                     "Age (day)": "age", #day
                     "Concrete compressive strength(MPa, megapascals) ":"concrete_compressive_strength"} #MPa

df.rename(columns=new_columns_names, inplace=True)

#%% Evaluation of model with default settings 

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.pipeline import make_pipeline

RSTATE = 11
TSIZE = 0.2
NSPLITS = 10

X, y = df[[c for c in df.columns if c !="concrete_compressive_strength"]], df["concrete_compressive_strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TSIZE, shuffle=True, random_state=RSTATE)

kf = KFold(n_splits=NSPLITS, shuffle=True)
kf_performance_metrics = {"MSE": [], "RMSE": [], "R2": []}


rfr_model = RandomForestRegressor(random_state=RSTATE)
dtr_model = DecisionTreeRegressor(random_state=RSTATE)

for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    
    
    print(f"Evaluating on split {i+1}")
    Xtrain, Xval = X_train.values[train_idx], X_train.values[val_idx]
    # print(f"Number of X samples in training data: {len(Xtrain)}")
    # print(f"Number of X samples in validation data: {len(Xval)}")
    
    ytrain, yval = y_train.values[train_idx], y_train.values[val_idx]
    # print(f"Number of y samples in training data: {len(ytrain)}")
    # print(f"Number of y samples in validation data: {len(yval)}")
    
    fitted_model = dtr_model.fit(Xtrain, ytrain.ravel())
    yvalpred = fitted_model.predict(Xval)
    
    kf_performance_metrics["MSE"] = mean_squared_error(yval, yvalpred)
    kf_performance_metrics["RMSE"] = np.sqrt(mean_squared_error(yval, yvalpred))
    kf_performance_metrics["R2"] = r2_score(yval, yvalpred)
    
    
    if False:
        plt.figure()
        plt.title(f"Fold {i+1}")
        plt.plot(yvalpred, "bo", label="Predicted values")
        plt.plot(yval, "ro", label="Ground truth")
        plt.grid()
        plt.legend()
        
    print(f"Validation MSE(fold n={i+1}): {kf_performance_metrics['MSE']}")
    print(f"Validation RMSE(fold n={i+1}): {kf_performance_metrics['RMSE']}")
    print(f"Validation R2(fold n={i+1}): {kf_performance_metrics['R2']}")
          
print(F"Mean MSE over all folds: {np.round(np.mean(kf_performance_metrics['MSE']), 3)}")
print(F"Mean RMSE over all folds: {np.round(np.mean(kf_performance_metrics['RMSE']), 3)}")
print(F"Mean R2 over all folds: {np.round(np.mean(kf_performance_metrics['R2']), 3)}")
# print(f"Number of X samples in training data: {len(X_train)}")
# print(f"Number of X samples in test data: {len(X_test)}")
fitted_model = dtr_model.fit(X_train.values, y_train.values.ravel())
model_prediction = fitted_model.predict(X_test.values)
print(f"DT model MSE on entire dataset: {mean_squared_error(y_test, model_prediction)}")
print(f"DT model RMSE on entire dataset: {np.sqrt(mean_squared_error(y_test, model_prediction))}")
print(f"DT model R2 on entire dataset: {r2_score(y_test, model_prediction)}")

if False:
    plt.figure()
    plt.title("Entire dataset")
    plt.plot(model_prediction, "bo", label="Predicted values")
    plt.plot(y_test.values, "ro", label="Ground truth")
    plt.grid()
    plt.legend()

def evaluate_model(X, y, target, model_hyperparams=None):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TSIZE, shuffle=True, random_state=RSTATE)
    
    kf = KFold(n_splits=NSPLITS, shuffle=True)
    kf_performance_metrics = {"MSE": [], "RMSE": [], "R2": []}
    
    
    if model_hyperparams is None:
        dtr_model = DecisionTreeRegressor(random_state=RSTATE)
    else:
        dtr_model = DecisionTreeRegressor(random_state=RSTATE, **model_hyperparams)
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        
        # Evaluating on split
        Xtrain, Xval = X_train.values[train_idx], X_train.values[val_idx]
        ytrain, yval = y_train.values[train_idx], y_train.values[val_idx]
        fitted_model = dtr_model.fit(Xtrain, ytrain.ravel())
        yvalpred = fitted_model.predict(Xval)
        
        kf_performance_metrics["MSE"].append(mean_squared_error(yval, yvalpred))
        kf_performance_metrics["RMSE"].append(np.sqrt(mean_squared_error(yval, yvalpred)))
        kf_performance_metrics["R2"].append(r2_score(yval, yvalpred))
    
    
    mean_target_over_folds = np.round(np.mean(kf_performance_metrics[target]), 3)
    return mean_target_over_folds
    

# plot_tree(fitted_model)

#%% Screening design with fractional factorial and perform ANOVA

factors = {"max_depth": [2, 20],
           "min_samples_split": [2, 50],
           "min_samples_leaf": [1, 20],
           "min_weight_fraction_leaf": [0, 0.2],
           "max_features": [1, 8],
           "max_leaf_nodes": [100, None],
           "ccp_alpha": [0, 0.2]}

factors = {
    "max_depth": {-1: 2, 1: 20},
    "min_samples_split": {-1: 2, 1: 50},
    "min_samples_leaf": {-1: 1, 1: 20},
    "min_weight_fraction_leaf": {-1: 0, 1: 0.2},
    "max_features": {-1: 1, 1: 8},
    "max_leaf_nodes": {-1: 100, 1: None},
    "ccp_alpha": {-1: 0, 1: 0.2}
}

experiment_design_template_path = "C:/Projects/MZIR/experiment_templates/fractional_factorial_design_screening.xlsx"
ffd_template = pd.read_excel(experiment_design_template_path)
ffd_template.set_index("Run Order", inplace=True)
# print(evaluate_model(X, y, "R2"))

# TODO do it yourself or check if works correctly
def map_real_values_to_template(template, factor_values):
    df_mapped = template.replace(factor_values)
    
    return df_mapped


def run_experiments(X, y, target, mapped_exp_des_template, n_repetitions=2):
    
    results_df = pd.DataFrame({"Run Order": [r for r in range(1, len(mapped_exp_des_template) + 1)]})
    results_df.set_index("Run Order", inplace=True)
    for repetition_idx in range(n_repetitions):
        rows = mapped_exp_des_template.to_dict(orient="records")
        repetition_result = np.zeros(shape=(len(mapped_exp_des_template), ))
        for i, run_hyperparams in enumerate(rows):
            repetition_result[i] = evaluate_model(X, y, target, run_hyperparams)
        
        results_df[f"{target}_{repetition_idx+1}"] = repetition_result
        
    return results_df

ffd_mapped_template = map_real_values_to_template(ffd_template, factors)
ffd_screening_results = run_experiments(X, y, target="MSE", mapped_exp_des_template=ffd_mapped_template, n_repetitions=1)

ffd_template_screening_results = pd.concat([ffd_template, ffd_screening_results], axis=1)
ffd_screening_results = pd.concat([ffd_mapped_template, ffd_screening_results], axis=1)

results_path = "C:/Projects/MZIR/experiments_results"
ffd_screening_filename = "FracFactDesign_screening_results.xlsx"
if False:
    ffd_screening_results.to_excel(results_path + "/" + ffd_screening_filename)
    
#%% ANOVA on ffd screening experiment
import seaborn as sns
from itertools import combinations
import statsmodels.api as sm
from statsmodels.formula.api import ols

def create_main_effects_plots(data, y, repetition_results:list):
    # Get all column names except for the response variable
    factors = [col for col in data.columns if col not in repetition_results]
    
    for fi, fname in enumerate(factors):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle(f"Main Effects Plot repetition_num: {y}")
        sns.pointplot(x=fname, y=y, data=data, ax=axs)
        axs.set_xlabel(fname)
        axs.set_ylabel(y)
        axs.grid()
        
    
    plt.tight_layout()
    
    
# 'Filtration_rate ~ T + CoF + RPM + T:CoF + T:RPM'
def generate_anova_formula(response_name, factor_names, max_interactions):
    if max_interactions <= 0 or max_interactions > len(factor_names):
        raise ValueError("'max_interactions' must be in range 0 < x <= len('factor_names')")
    
    model_formula = ""
    model_formula += response_name + " ~ "
    model_formula += " + ".join(factor_names)

    combined_factors_at_interaction_num = []
    for interaction_num in range(1, max_interactions+1):
        for combo in combinations(factor_names, interaction_num):
            combined_factors_at_interaction_num.append(":".join(combo))
    
    model_formula += " + " + " + ".join(combined_factors_at_interaction_num)
    return model_formula

if False:
    create_main_effects_plots(ffd_template_screening_results, "MSE_1", ["MSE_1"])

anova_model_formula = generate_anova_formula("MSE_1", factors.keys(), max_interactions=2)
model = ols(anova_model_formula, data=ffd_template_screening_results).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)