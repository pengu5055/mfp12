"""
Plot gathered data from BoostMe object and save to file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Settings ---
# Set to True to run the raw performance test
rawP = False
# Set to True to run the hyperparameter search for learning rate and max depth
lr_md_search = False

lr_sweep = True

md_sweep = False

# --- Plot Settings ---

c_set = ['#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a']


# --- Plot Raw Performance ---
if rawP:
    # Load the data
    raw_performance_CPU = pd.read_csv("results/raw_performance_CPU.csv")
    raw_performance_GPU = pd.read_csv("results/raw_performance_GPU.csv")
    
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(raw_performance_CPU["Trees"], raw_performance_CPU["Time"], label="CPU", c=c_set[4])
    # Due to a bug the times in GPU data are negative, so we need to flip the sign
    plt.plot(raw_performance_GPU["Trees"], -raw_performance_GPU["Time"], label="GPU", c=c_set[6])
    plt.xlabel("Number of Trees")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs. Number of Trees @ LR=0.1, MD=6")
    plt.legend()
    plt.grid(c=c_set[2], alpha=0.4)
    plt.savefig("images/raw_performance.png")
    plt.show()

# --- Plot Learning Rate and Max Depth Search ---
if lr_md_search:
    # Ranges of hyperparameters
    learning_rates = np.linspace(0.001, 0.1, 16)
    max_depths = np.arange(1, 16, 1)

    # Load the data
    data = pd.read_csv("results/lr_md_search_GPUv2.csv").to_numpy()

    print(data)
    
    # Get the AUC values
    grid_auc = np.empty((len(learning_rates), len(max_depths)))
    grid_t = np.empty((len(learning_rates), len(max_depths)))
    for i, lr in enumerate(learning_rates):
        for j, md in enumerate(max_depths):
            print()
            grid_auc[i, j] = (data.iloc[i, j])[3]
            #grid_t[i, j] = (data.iloc[i, j])[2]

    
    # Plot the data
    plt.figure(figsize=(10, 5))

    # Create the heatmap
    plt.imshow(grid_auc, cmap="viridis", origin="lower", extent=[1, 15, 0.001, 0.1], aspect="auto")
    plt.colorbar()
    plt.xlabel("Max Depth")
    plt.ylabel("Learning Rate")
    plt.title("AUC vs. Learning Rate and Max Depth")
    plt.savefig("images/lr_md_search_GPU.png")
    plt.show()

# Plot the learning rate sweep
if lr_sweep:
    # Load the data
    with np.load("results/lr_sweepv2.npz") as data:
        aucs = data["aucs"]
        times = data["times"]
        lrs = data["learning_rates"]
    
    # Plot the data
    plt.figure(figsize=(10, 5))

    # Create the heatmap
    plt.plot(lrs, aucs, c=c_set[4])
    plt.xlabel("Learning Rate")
    plt.ylabel("AUC")
    plt.title("AUC vs. Learning Rate @ MD=6, Trees=100")
    plt.grid(c=c_set[2], alpha=0.4)
    plt.savefig("images/lr_sweepv2.png")
    plt.show()

if md_sweep:
    # Load the data
    with np.load("results/md_sweep.npz") as data:
        aucs = data["aucs"]
        times = data["times"]
        mds = data["max_depths"]
    
    # Plot the data
    plt.figure(figsize=(10, 5))

    # Create the heatmap
    plt.plot(mds, aucs, c=c_set[4])
    plt.xlabel("Max Depth")
    plt.ylabel("AUC")
    plt.title("AUC vs. Max Depth @ LR=0.1, Trees=100")
    plt.grid(c=c_set[2], alpha=0.4)
    plt.savefig("images/md_sweep.png")
    plt.show()