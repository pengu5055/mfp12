"""
Main loop for the neural network model.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nn import WhereAreThouHiggs

filename = "data/higgs-parsed/higgs-parsed.h5"
NN = WhereAreThouHiggs(filename)

# Create the model
NN._create_model()
NN.model.to(NN.device)
NN.train_model()
NN.plot_loss()