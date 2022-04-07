#!/usr/bin/env python
# coding: utf-8

# Functions for preprocessing data

# Librairies
# Data Manipulation
import pandas as pd


def get_sample_df(path, ratio) :
    """ 
    Function to get a sample from a dataframe
    """
    # Import dataset
    df = pd.read_parquet(path)
    # Let's take a sample of the data.
    sample_df = df.sample(frac=ratio, random_state=1)
    # Sort by time_id and investment_id to get data in order 
    # and reset index
    sample_df = sample_df.sort_values(
        ["time_id", "investment_id"], ascending=[True, True]
    ).reset_index(drop=True)
    
    return sample_df

def X_y(df, col_X, col_y) :
    """
    Function to get X and y arrays from a dataframe
    """
    X = df[col_X].values
    y = df[col_y].values
    return X,y

def split_data(X,y,index) :
    """
    Function to split X and y into trainset and testset
    """
    X_train = X[:index]
    X_test = X[index:]

    y_train = y[:index]
    y_test = y[index:]
    
    return X_train, y_train, X_test, y_test

def reshape_X(X) :
    """
    Function to reshape X for convolution 1D
    """
    sample_size = X.shape[0]  # number of samples in train set
    time_steps = X.shape[1]  # number of features in train set
    input_dimension = 1
    X_reshaped = X.reshape(sample_size, time_steps, input_dimension)
    
    return X_reshaped
