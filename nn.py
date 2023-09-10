"""
Same as with the BDT algorithm, we will use the PyTorch library to 
process the Higgs dataset and find the Higgs boson.
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import Callable

# --- Global Variables ---
# Set the number of epochs to train for
EPOCHS = 100
# Set the batch size
BATCH_SIZE = 1000
# Set the learning rate
LEARNING_RATE = 0.001
# Set the number of hidden layers
HIDDEN_LAYERS = 3
# Set the number of nodes per hidden layer
HIDDEN_NODES = 100
# Set the dropout rate
DROPOUT_RATE = 0.5
# Cmap
c_set = ['#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a']

class WhereAreThouHiggs():
    def __init__(self,
                 dataset_path: str,
                 ) -> None:
        # --- Global Variables ---
        self.dataset_path = dataset_path
        # Set the device to use
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # --- Load Data ---
        # Load data from HDF5
        self.rawdata = self._load_data()

    def _internal_function_timer(func: Callable):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Function {func.__name__} took {end - start} seconds to run.")
            return result, end - start
        return wrapper


    def _load_data(self):
        # Load data from HDF5
        # Remove the .h5 extension if it was already given
        self.dataset_path = self.dataset_path.removesuffix('.h5')
        self.dataset_path = self.dataset_path + '.h5'
        print(f"Loading data from {self.dataset_path}...")
        self.rawdata = pd.HDFStore(self.dataset_path,'r')
        self.data_features=self.rawdata['feature_names'].to_numpy()[1:]

        return self.rawdata
    
    def _split_feat_label(self, rawdata):
        data_y = rawdata['hlabel'] 
        data_x = rawdata.drop(['hlabel'], axis=1)

        # Scale to [0,1] range
        data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())

        return data_x, data_y
    
    def create_model(self):
        # Create the model
        self.model = nn.Sequential(
            nn.Linear(self.data_features.shape[0], self.data_features.shape[0]), # Input layer
            nn.ReLU(),
            nn.Linear(self.data_features.shape[0], self.data_features.shape[0]), # Hidden layer
            nn.ReLU(),
            nn.Linear(self.data_features.shape[0], 1), # Output layer
            nn.Sigmoid()
        )

        # Move the model to the device
        self.model.to(self.device)

        return self.model
    
    def _data_split(self, batch_size=BATCH_SIZE):
        # --- Split into Training and Testing ---
        # Split data into training and testing sets
        tr_x, tr_y = self._split_feat_label(self.rawdata['train'])
        self.vl_x, self.vl_y = self._split_feat_label(self.rawdata['valid'])
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(tr_x, tr_y, test_size=0.1)

        # --- Create the Data Loaders ---
        # Create the training and testing data loaders
        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(self.train_x.to_numpy()), torch.tensor(self.train_y.to_numpy())), batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(self.test_x.to_numpy()), torch.tensor(self.test_y.to_numpy())), batch_size=batch_size, shuffle=True)

        return self.train_loader, self.test_loader
    
    @_internal_function_timer
    def train_model(self, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
        # Split the data
        self._data_split(batch_size=batch_size)

        # Create the optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Create the loss function
        self.loss_fn = nn.MSELoss()

        # Train the model
        self.train_losses = []
        self.test_losses = []
        for epoch in range(epochs):
            # Train
            train_loss = 0
            for i, (x, y) in enumerate(self.train_loader):
                # Move data to device
                x = x.to(self.device)
                y = y.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                self.y_hat = self.model(x.float())
                loss = self.loss_fn(self.y_hat, y.float().unsqueeze(1))

                # Backward pass
                self.optimizer.step()
                loss.backward()

                # Add the loss
                train_loss += loss.item()
                # train_losses.append(train_loss)

            # Test
            test_loss = 0
            for i, (x, y) in enumerate(self.test_loader):
                # Move data to device
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                self.y_hat = self.model(x.float())
                loss = self.loss_fn(self.y_hat, y.float().unsqueeze(1))

                # Add the loss
                test_loss += loss.item()

            # Save the losses
            self.train_losses.append(train_loss / len(self.train_loader))
            self.test_losses.append(test_loss / len(self.test_loader))

            # Print the loss
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss / len(self.train_loader)} | Test Loss: {test_loss / len(self.test_loader)}")

        #return self.train_losses, self.test_losses NOTE add back in if needed

    def plot_loss(self):
        # Plot the losses
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss", c=c_set[4])
        plt.plot(self.test_losses, label="Test Loss", c=c_set[6])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epoch")
        plt.legend()
        plt.savefig("images/losses.png")
        plt.show()

    def performance(self):
        # Get predictions
        preds = self.model(torch.tensor(self.vl_x.to_numpy()).to(self.device).float())

        # Get the ROC AUC score
        auc = roc_auc_score(torch.tensor(self.vl_y.to_numpy()), preds.detach().cpu().numpy())

        print(f"AUC Score: {auc}")
        return auc

    def plot_model(self):
        from torchviz import make_dot

        make_dot(self.yhat, params=dict(list(self.model.named_parameters()))).render("nn_torchviz", format="png")
    
