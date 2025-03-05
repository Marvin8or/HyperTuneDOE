import doe_utils as du
import numpy as np
import pandas as pd

results = pd.DataFrame(
    {"X1": [-1, 1, -1, 1], "X2": [-1, -1, 1, 1], "Y": [100, 200, 150, 250]}
)

du.create_main_effects_plots(results, "Y", figsize=du.FIGSIZE_HALF)
du.create_interaction_plots(results, "Y", figsize=du.FIGSIZE_HALF)

results = pd.DataFrame(
    {"X1": [-1, 1, -1, 1], "X2": [-1, -1, 1, 1], "Y": [100, 200, 250, 150]}
)
du.create_interaction_plots(results, "Y", figsize=du.FIGSIZE_HALF)
