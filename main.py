"""
This is the main loop of the program. It is responsible for running the
CatBoost model and saving the results of timings and performance to a
file.
"""

import pandas as pd
import numpy as np
from boostme import BoostMe

# --- Create the BoostMe Object ---
# Create the BoostMe object
dataset_path = "data/higgs-parsed/higgs-parsed.h5"
bm = BoostMe(dataset_path)

# --- Hyperparameters ---
# Hyperparameters to search over (task 2.)
learning_rates = [0.1, 0.01, 0.001]
max_depths = [6, 8, 10]
l2_leaf_reg = [1, 3, 5]
random_strengths = [0.1, 0.5, 1]
bagging_temperatures = [0.1, 0.5, 1]


# --- Run the Model ---
# First lets measure raw performance
trees = np.arange(100, 1000, 100)
results = []
for t in trees:
    print(f"Training with {t} trees...")
    m, time, = bm.train(max_trees=t, verbose=False)
    auc = bm.performance()
    results.append([t, time, auc])

# Save the results
results = pd.DataFrame(results, columns=["Trees", "Time", "AUC"])
results.to_csv("results/raw_performance_GPU.csv", index=False)
