import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Settings ---
# Set to True to run the raw performance test
rawP = False

# Set to True to run the hyperparameter search for learning rate and max depth
lr_sweep = True

c_set = ['#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a']


# --- Plot Raw Performance ---
if rawP:
    # Load the data
    with np.load("results/NNraw_performance.npz", allow_pickle=True) as data:
        epoch_range = data["arr_0"]
        auc = data["arr_1"]
        time = data["arr_2"]
    
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, time, label="GPU", c=c_set[4])
    plt.xlabel("Number of Trees")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs. Number of Trees @ LR=0.1, MD=6")
    plt.legend()
    plt.grid(c=c_set[2], alpha=0.4)
    plt.savefig("images/NNraw_performance.png")
    plt.show()

# --- Plot Learning Rate and Max Depth Search ---
if lr_sweep:
    with np.load("results/NNlr_sweep.npz", allow_pickle=True) as data:
        learning_rates = data["arr_0"]
        aucs = data["arr_1"]
        times = data["arr_2"]
    
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, aucs, label="AUC", c=c_set[4])

    plt.xlabel("Learning Rate")
    plt.ylabel("AUC")
    plt.title("AUC vs. Learning Rate @ Epochs=10")
    plt.legend()
    plt.grid(c=c_set[2], alpha=0.4)
    plt.savefig("images/NNlr_sweep.png")
    plt.show()


