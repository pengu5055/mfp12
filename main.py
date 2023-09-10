"""
This is the main loop of the program. It is responsible for running the
CatBoost model and saving the results of timings and performance to a
file.
"""

import pandas as pd
import numpy as np
from boostme import BoostMe

# --- Settings ---
# Set to True to run the raw performance test
rawP = False
# Set to True to run the hyperparameter search for learning rate and max depth
lr_md_search = False
# Set to True to run the learning rate sweep
lr_sweep = True

# --- Plot Settings ---
c_set = ['#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a']

# --- Create the BoostMe Object ---
# Create the BoostMe object
dataset_path = "data/higgs-parsed/higgs-parsed.h5"
bm = BoostMe(dataset_path)

# --- Hyperparameters ---
# Hyperparameters to search over (task 2.)
# learning_rates = [0.1, 0.01, 0.001]
learning_rates = np.linspace(0.001, 0.1, 16)
# max_depths = [6, 8, 10]
max_depths = np.arange(1, 16, 1)
l2_leaf_reg = [1, 3, 5]
random_strengths = [0.1, 0.5, 1]
bagging_temperatures = [0.1, 0.5, 1]


# --- Run the Model ---
# First lets measure raw performance
if rawP:
    trees = np.arange(100, 1000, 10)
    results = []
    for t in trees:
        print(f"Training with {t} trees...")
        m, time, = bm.train(max_trees=t, task_type="CPU", verbose=False)
        auc = bm.performance()
        results.append([t, time, auc])
    # Save the results
    results = pd.DataFrame(results, columns=["Trees", "Time", "AUC"])
    results.to_csv("results/raw_performance.csv", index=False)

# Now lets run the hyperparameter search for learning rate and max depth
if lr_md_search:
    results = np.empty((len(learning_rates), len(max_depths)), dtype=object)
    for i, lr in enumerate(learning_rates):
        for j, md in enumerate(max_depths):
            print(f"Training with learning rate {lr} and max depth {md}...")
            m, time, = bm.train(max_trees=100, task_type="GPU", verbose=False, learning_rate=lr, max_depth=md)
            auc = bm.performance()
            results[i, j] = [lr, md, time, auc]
    # Save the results
    results = pd.DataFrame(results)
    results.to_csv("results/lr_md_search_GPUv2.csv", index=False)

if lr_sweep:
    learning_rates = np.linspace(0.001, 0.1, 100)
    aucs = []
    times = []
    for i, lr in enumerate(learning_rates):
        print(f"Training with learning rate {lr}...")
        m, time, = bm.train(max_trees=100, task_type="GPU", verbose=False, learning_rate=lr)
        auc = bm.performance()
        aucs.append(auc)
        times.append(time)

    # Save the results
    np.savez("results/lr_sweep.npz", learning_rates=learning_rates, aucs=aucs, times=times)


