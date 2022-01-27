# Step 3. Train Model
# Sample Python script designed to train a K-Neighbors classification
# model using the Scikit-Learn library.

from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import os
import argparse
import shutil

import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import joblib
from numpy.random import seed


# Parse input arguments
parser = argparse.ArgumentParser("Train classification model")
parser.add_argument('--train_to_evaluate_pipeline_data', dest='train_to_evaluate_pipeline_data', required=True)
parser.add_argument('--target_column', type=str, required=True)

args, _ = parser.parse_known_args()
train_to_evaluate_pipeline_data = args.train_to_evaluate_pipeline_data
target_column = args.target_column

# Get current run
current_run = Run.get_context()

#G et associated AML workspace
ws = current_run.experiment.workspace

# Read input dataset to pandas dataframe
X_train_dataset = current_run.input_datasets['Training_Data']
X_train = X_train_dataset.to_pandas_dataframe().astype(np.float64)
X_test_dataset = current_run.input_datasets['Testing_Data']
X_test = X_test_dataset.to_pandas_dataframe().astype(np.float64)

# Split into X and y 
y_train = X_train[[target_column]]
y_test = X_test[[target_column]]

X_train = X_train.drop(target_column, axis=1)
X_test = X_test.drop(target_column, axis=1)

################################# MODIFY #################################

# The intent of this block is to scale data appropriately and train
# a clustering model. Any normalizaton and training approach can be used.
# Serialized scalers/models can be passed forward to subsequent pipeline
# steps as PipelineData using the syntax below. Additionally, for 
# record-keeping, it is recommended to log performance metrics 
# into the current run.

# Convert to expected format
import numpy as np
X_train_arr = X_train.to_numpy()
X_train_arr = X_train_arr.reshape(X_train_arr.shape[0], X_train_arr.shape[1], 1)

X_test_arr = X_test.to_numpy()
X_test_arr = X_test_arr.reshape(X_test_arr.shape[0], X_test_arr.shape[1], 1)

from tslearn.preprocessing import TimeSeriesScalerMeanVariance

scaler = TimeSeriesScalerMeanVariance().fit(X_train_arr)
X_train = scaler.transform(X_train_arr)
X_test = scaler.transform(X_test_arr)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=4, verbose=True, random_state=seed)
model = km.fit(X_train)
y_pred = model.predict(X_train)

# Save model to outputs for record-keeping
os.makedirs('./outputs', exist_ok=True)
from joblib import dump, load
dump(model, './outputs/model.pkl')
dump(scaler, './outputs/scaler.pkl')

# Save model to pipeline_data for use in evaluation/registration step
os.makedirs(train_to_evaluate_pipeline_data, exist_ok=True)
dump(model, os.path.join(train_to_evaluate_pipeline_data, 'model.pkl'))
dump(scaler, os.path.join(train_to_evaluate_pipeline_data, 'scaler.pkl'))

# Generate predictions
plt.figure()
for yi in range(4):
    plt.subplot(4, 1, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(model.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

current_run.log_image('Plot', plot=plt)

##########################################################################