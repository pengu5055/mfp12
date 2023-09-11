"""
Main loop for the neural network model.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nn import WhereAreThouHiggs
import torch

filename = "data/higgs-parsed/higgs-parsed.h5"
NN = WhereAreThouHiggs(filename)
NN.create_model()

# --- Settings ---
# Set to True to run the raw performance test
rawP = False

# Set to True to run the hyperparameter search for learning rate and max depth
lr_sweep = False

# Set to True to plot signal vs. background
plot_sig_bk = True

# --- Run the Model ---
# First lets measure raw performance
if rawP:
    epoch_range = np.arange(1, 21, 1)
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

# Now lets run the hyperparameter search for learning rate and max depth
if lr_sweep:
    learning_rates = np.linspace(0.001, 2, 30)
    aucs = []
    times = []
    for i, lr in enumerate(learning_rates):
        print(f"Training with learning rate {lr}...")
        NN = WhereAreThouHiggs(filename)
        NN.create_model()
        m, time, = NN.train_model(epochs=10, learning_rate=lr)
        auc = NN.performance()
        aucs.append(auc)
        times.append(time)

    # Save the results
    np.savez("results/NNlr_sweep.npz", learning_rates=learning_rates, aucs=aucs, times=times)

if plot_sig_bk:
    train = False
    plot = True
    if train:
        m, time = NN.train_model(epochs=100, learning_rate=0.1)
        
        with torch.inference_mode():
            model = NN.model.to("cpu")
            y_hat = model(torch.tensor(NN.vl_x.to_numpy()))
        y_hat = y_hat.detach().numpy()
        np.savez("results/NNsig_bk_100.npz", y_hat, NN.vl_y.to_numpy())
    if plot:
        with np.load("results/NNsig_bk.npz", allow_pickle=True) as data:
            y_hat = data["arr_0"]
            vl_y = data["arr_1"]
        
        
        NN.plot_score(vl_y, y_hat, "images/NNsig_bk_10.png")

