"""
Plot gathered data from BoostMe object and save to file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Plot Raw Performance ---
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
