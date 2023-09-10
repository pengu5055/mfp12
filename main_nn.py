"""
Main loop for the neural network model.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nn import WhereAreThouHiggs

filename = "data/higgs-parsed/higgs-parsed.h5"
NN = WhereAreThouHiggs(filename)
NN.create_model()

# --- Settings ---
# Set to True to run the raw performance test
rawP = True


# --- Run the Model ---
# First lets measure raw performance
if rawP:
    epoch_range = np.arange(1, 100, 1)
    aucs = []
    time = []
    for e in epoch_range:
        print(f"Training with {e} epochs...")
        t = NN.train_model(epochs=e,learning_rate=0.1)
        auc = NN.performance()
        time.append(t)
        aucs.append(auc)
    # Save the results
    np.savez("results/NNraw_performance.npz", epoch_range, auc, time)