"""
Create a class to contain all that is required to run the 
ML model on the Higgs data once at given hyperparameters.
This is so we can do a grid search over the hyperparameters
and run the model many times.
"""
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from typing import Callable

c_set = ['#fff7f3','#fde0dd','#fcc5c0','#fa9fb5','#f768a1','#dd3497','#ae017e','#7a0177','#49006a']

class BoostMe():
    def __init__(self,
                 dataset_path: str,
                 ) -> None:
        # --- Global Variables ---
        self.dataset_path = dataset_path
        # --- Load Data ---
        # Load data from HDF5
        self.rawdata = self._load_data()
        self.feature_list = self.rawdata["feature_names"].to_numpy()[1:]

        # --- Split into Training and Testing ---
        # Split data into training and testing sets
        tr_x, tr_y = self._split_feat_label(self.rawdata['train'])
        self.vl_x, self.vl_y = self._split_feat_label(self.rawdata['valid'])
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(tr_x, tr_y, test_size=0.1)

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
        self.dataset_path = self.dataset_path + '.h5' # Add the .h5 extension to make sure it is there

        print(f"Loading data from {self.dataset_path}...")
        self.rawdata = pd.HDFStore(self.dataset_path,'r')
        return self.rawdata
    
    def _split_feat_label(self, rawdata):
        data_y = rawdata['hlabel'] 
        data_x = rawdata.drop(['hlabel'], axis=1)

        # Scale to [0,1] range
        data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())

        return data_x, data_y
    
    @_internal_function_timer
    def train(self, max_trees=1000, task_type="GPU", eval_metric="AUC", verbose=False):
        # --- Create the CatBoost Pool ---
        # Create the training and testing pools
        train_pool = Pool(data=self.train_x.to_numpy(), label=self.train_y.to_numpy(),
                          feature_names=self.feature_list.tolist())
        test_pool = Pool(data=self.test_x.to_numpy(), label=self.test_y.to_numpy(), 
                         feature_names=self.feature_list.tolist())

        # --- CatBoost Classifier ---
        # Create the CatBoost classifier
        self.model = CatBoostClassifier(iterations=max_trees, 
                                task_type=task_type,
                                learning_rate=0.1,  
                                loss_function='Logloss',
                                eval_metric=eval_metric,
                                use_best_model=True,
                                verbose=verbose)

        # Train the model
        self.model.fit(train_pool, eval_set=test_pool, plot=False)

        return self.model
    
    def performance(self):
        # --- Performance ---
        # Get predictions
        preds = self.model.predict(self.vl_x)

        # Get the ROC AUC score
        auc = roc_auc_score(self.vl_y, preds)

        print(f"AUC Score: {auc}")
        return auc

    def plot_roc(self):
        # --- Plotting ---
        plt.figure()
        plt.plot(self.model.evals_result_['learn']['Logloss'], label='Training Loss', c=c_set[4])
        plt.plot(self.model.evals_result_['validation']['Logloss'], label='Testing Loss', c=c_set[6])
        plt.xlabel('Iteration')
        plt.ylabel('Logloss')
        plt.title('CatBoost Loss')
        plt.legend()
        plt.show()