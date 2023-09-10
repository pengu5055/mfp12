"""
Process the Higgs data to do a ML classification
and find the Higgs boson signal in the data
"""
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

# BPK flavor data processing
import data_higgs as dh

# --- Global Variables ---
BATCH_SIZE = 1000

def split_feat_label(rawdata):
    data_y = rawdata['hlabel'] 
    data_x = rawdata.drop(['hlabel'], axis=1)

    # Scale to [0,1] range
    data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())

    return data_x, data_y

if __name__ == "__main__":
    # --- Load Data ---
    # Load data from CSV file
    rawdata = dh.load_data()
    feature_list = rawdata["feature_names"].to_numpy()[1:]

    # --- Split into Training and Testing ---
    # Split data into training and testing sets
    tr_x, tr_y = split_feat_label(rawdata['train'])
    vl_x, vl_y = split_feat_label(rawdata['valid'])
    train_x, test_x, train_y, test_y = train_test_split(tr_x, tr_y, test_size=0.1)

    # --- Create the CatBoost Pool ---
    # Create the training and testing pools
    train_pool = Pool(data=train_x.to_numpy(), label=train_y.to_numpy(), feature_names=feature_list.tolist())
    test_pool = Pool(data=test_x.to_numpy(), label=test_y.to_numpy(), feature_names=feature_list.tolist())


    # --- CatBoost Classifier ---
    # Create the CatBoost classifier
    eval_metric = "AUC"
    task_type = "GPU"
    max_trees = 1000
    model = CatBoostClassifier(iterations=max_trees, 
                               task_type=task_type,
                               learning_rate=0.1,  
                               loss_function='Logloss',
                               eval_metric=eval_metric,
                               use_best_model=True,
                               verbose=True)

    # Train the model
    model.fit(train_pool, eval_set=test_pool, plot=True)

    # --- Performance ---
    # Get predictions
    preds = model.predict(vl_x)

    # Get the ROC AUC score
    auc = roc_auc_score(vl_y, preds)

    print(f"AUC Score: {auc}")

    # --- Plotting ---
    # Plot the ROC curve
    plt.figure()
    plt.plot(model.evals_result_['validation']['Logloss'], label='Training Loss')
    plt.plot(model.evals_result_['validation']['Logloss'], label='Testing Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Logloss')
    plt.title('CatBoost Loss')
    plt.legend()
    plt.show()
