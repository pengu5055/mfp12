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
lr_md_search = True



# --- Plot Raw Performance ---
if rawP:
    # Load the data
    raw_performance_CPU = pd.read_csv("results/raw_performance_CPU.csv")
    raw_performance_GPU = pd.read_csv("results/raw_performance_GPU.csv")
    
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(raw_performance_CPU["Trees"], raw_performance_CPU["Time"], label="CPU")
    # Due to a bug the times in GPU data are negative, so we need to flip the sign
    plt.plot(raw_performance_GPU["Trees"], -raw_performance_GPU["Time"], label="GPU")
    plt.xlabel("Number of Trees")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs. Number of Trees")
    plt.legend()
    plt.savefig("images/raw_performance.png")
    plt.show()

# --- Plot Learning Rate and Max Depth Search ---
if lr_md_search:
    # Ranges of hyperparameters
    learning_rates = np.linspace(0.001, 0.1, 16)
    max_depths = np.arange(1, 16, 1)

    # Load the data
    lr, md, t, acu = np.genfromtxt("results/lr_md_search_GPUv2.csv", delimiter=",", skip_header=1, unpack=True)
    print(lr, md, t, acu)

    # Extract the ACU values as on a grid defined by the hyperparameters
    grid_acu = np.empty((len(lr), len(md)))
    for i in range(len(lr)):
        for j in range(len(md)):
            grid_acu[i, j] = acu[i]

    grid_t = np.empty((len(lr), len(md)))
    for i in range(len(lr)):
        for j in range(len(md)):
            grid_acu[i, j] = t[i//len(learning_rates)**2 + j]
            

    # Plot the data
    plt.figure(figsize=(10, 5))

    # Create the heatmap
    plt.imshow(grid_acu, cmap="viridis", origin="lower", extent=[1, 15, 0.001, 0.1], aspect="auto")
    plt.colorbar()
    plt.xlabel("Max Depth")
    plt.ylabel("Learning Rate")
    plt.title("AUC vs. Learning Rate and Max Depth")
    plt.savefig("images/lr_md_search_GPU.png")
    plt.show()


