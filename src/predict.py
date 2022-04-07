#!/usr/bin/env python
# coding: utf-8

# Ubiquant market prediction : Modelisation with best model
# Kaggle competition 
# https://www.kaggle.com/competitions/ubiquant-market-prediction/overview.

# Librairies
# Data Manipulation

import pandas as pd

# Create/Get files content
import joblib

# scipy tools
from scipy.stats.stats import pearsonr

# Tensorflow
from tensorflow import keras


# load data
X_test_reshaped = joblib.load("./Data/X_test_reshaped.joblib")
y_test = joblib.load("./Data/y_test.joblib")

model_file_name = "./Models/Conv1D.tf"
model_conv1D = keras.models.load_model("./Models/")

# Model evaluation
model_conv1D.load_weights(model_file_name)
model = "Conv1D"
[loss_test, mae_test, rmse_test] = model_conv1D.evaluate(
    X_test_reshaped, y_test, verbose=1
)

# Get pearson score
pearson_score = pearsonr(
    model_conv1D.predict(X_test_reshaped).reshape(-1), y_test)[0]

# Results in dataframe
df_test = pd.DataFrame(
    [[model, loss_test, mae_test, rmse_test, pearson_score]],
    columns=["model", "loss_test", "mae_test", "rmse_test", "pearson_score"],
)

# Display result
display(df_test)