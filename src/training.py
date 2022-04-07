#!/usr/bin/env python
# coding: utf-8

# Ubiquant market prediction : Modelisation with best model
# Kaggle competition 
# https://www.kaggle.com/competitions/ubiquant-market-prediction/overview.

# Librairies
# Data Manipulation
import numpy as np
import pandas as pd

# Create/Get files content
import os
import joblib

# Tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from preprocessing import *

# We use parquet version of the dataset thanks to this kernel 
# https://www.kaggle.com/code/camilomx/parquet-format-quickstart.

# Import dataset
path = "./train_low_mem.parquet"
ratio = 0.05

sample_df = get_sample_df(path, ratio)
features = [f"f_{i}" for i in range(300)]

X, y = X_y(sample_df,features, "target")

X_train, y_train, X_test, y_test = split_data(X, y, 140000)

X_train_reshaped = reshape_X(X_train)
X_test_reshaped = reshape_X(X_test)

# Save data
data_dir_path = "./Data"
os.makedirs(data_dir_path, exist_ok=True)
joblib.dump(X_train_reshaped, "./Data/X_train_reshaped.joblib")
joblib.dump(X_test_reshaped, "./Data/X_test_reshaped.joblib")
joblib.dump(y_train, "./Data/y_train.joblib")
joblib.dump(y_test, "./Data/y_test.joblib")


# Model construction
# Inspired by https://www.kaggle.com/code/ghostcxs/prediction-including-spatial-info-with-conv1d
def build_conv1D_model(train_data_reshaped):
    
    """
    Function to create neuronal networks with conv1D
    
    Arguments :
    - train_data_reshaped : training data with correct shape.
    
    Return :
    - model : compiled model with all layers.
    """
    
    n_timesteps = train_data_reshaped.shape[1]  
    n_features = train_data_reshaped.shape[2]  
    
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_timesteps, n_features)))

    # Convolution 1
    model.add(
        keras.layers.Conv1D(
            filters=64,
            kernel_size=4,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv1D_1",
        )
    )
    
    # Dropout
    model.add(keras.layers.Dropout(0.5))

    # Convolution 2
    model.add(
        keras.layers.Conv1D(
            filters=16, kernel_size=8, strides=3, activation="relu", name="Conv1D_2"
        )
    )

    # Convolution 3
    model.add(
        keras.layers.Conv1D(
            filters=16, kernel_size=3, strides=1, activation="relu", name="Conv1D_3"
        )
    )

    # Convolution 4
    model.add(
        keras.layers.Conv1D(
            filters=32,
            kernel_size=4,
            strides=3,
            padding="same",
            activation="relu",
            name="Conv1D_4",
        )
    )

    # Convolution 5
    model.add(
        keras.layers.Conv1D(
            filters=64,
            kernel_size=4,
            strides=4,
            padding="same",
            activation="relu",
            name="Conv1D_5",
        )
    )

    # Flatten 
    model.add(keras.layers.Flatten())
    
    # Dense layers
    model.add(keras.layers.Dense(64, activation="relu", name="Dense_2",))
    model.add(keras.layers.Dense(32, activation="relu", name="Dense_3",))
    model.add(keras.layers.Dense(n_features, name="Dense_4"))
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(0.001)
    
    # RMSE
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    
    # Model compilation
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", rmse])
    
    # Summary
    model.summary()
    
    return model

# Create model 
model_conv1D = build_conv1D_model(X_train_reshaped)

# File to save model
models_dir_path = "./Models"
os.makedirs(models_dir_path, exist_ok=True)
model_file_name = models_dir_path + "/Conv1D.tf"

# Checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=model_file_name,
    monitor="val_rmse",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

# Earlystop
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_rmse", patience=5, verbose=1, restore_best_weights=True
)

# Model training
EPOCHS=30
BATCH_SIZE = 4096
history = model_conv1D.fit(x=X_train_reshaped, 
                             y=y_train, 
                             epochs=EPOCHS, 
                             shuffle= False,
                             batch_size=BATCH_SIZE, 
                             validation_split=0.2, 
                             callbacks=[checkpoint, early_stop], 
                             verbose=1)

# Convert the history.history dict to a pandas DataFrame
hist_df = pd.DataFrame(history.history)

# Save to csv
hist_csv_file = "./Models/history.csv"
with open(hist_csv_file, mode="w") as f:
    hist_df.to_csv(f)

# Save model
model_conv1D.save(models_dir_path)