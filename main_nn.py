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
plot_sig_bk = False

# Set to True to plot the ROC curve
roc = True

c_set = ['#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a']


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

if roc:
    # Load the data
    with np.load("results/NNsig_bk.npz", allow_pickle=True) as data:
            y_hat1 = data["arr_0"]
            vl_y1 = data["arr_1"]
    
    fpr1, tpr1, t1 = NN.plot_roc(vl_y1, y_hat1, "images/NN_roc.png")

    # Load the data
    with np.load("results/NNsig_bk_100.npz", allow_pickle=True) as data:
            y_hat2 = data["arr_0"]
            vl_y2 = data["arr_1"]
    
    fpr2, tpr2, t2 = NN.plot_roc(vl_y2, y_hat2, "images/NN_roc_100.png")

    # Load the data

    with np.load("results/NNsig_bk_500.npz", allow_pickle=True) as data:
            y_hat3 = data["arr_0"]
            vl_y3 = data["arr_1"] 
        
    fpr3, tpr3, t3 = NN.plot_roc(vl_y3, y_hat3, "images/NN_roc_500.png")
    plt.plot(fpr1, tpr1, label='Epochs = 10', c=c_set[4])
    plt.plot(fpr2, tpr2, label='Epochs = 100', c=c_set[6])
    plt.plot(fpr3, tpr3, label='Epochs = 500', c=c_set[8])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(c=c_set[2], alpha=0.4)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c=c_set[-1], alpha=0.4, ls="--")
    plt.legend()
    plt.savefig("images/NN_roc.png")
    plt.show()

