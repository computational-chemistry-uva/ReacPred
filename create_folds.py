#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:14:37 2024

@author: korotkevich
"""

import pandas as pd
import os

def create_folds(path, k_folds, random_state = 42):
    
    '''
    This function creates a number of files with data
    used for training and testing a model for k-fold cross-validation.
    Takes path to the original data .csv file,
    desired number of folds (k_folds),
    random_state for data shuffling (optional)
    Creates folder with files for training and testing for each of k folds
    
    '''
    
    #Read the database to a pandas df
    df = pd.read_csv(path)
    #Shuffle rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    df_size = len(df)
    indices = list(range(df_size))
    fold_size = df_size // k_folds
    fold_losses = []
    
    # Check if the directory with folds already exists
    directory_name = 'Folds'
    if not os.path.exists(directory_name):
        # Create the directory
        os.makedirs(directory_name)
    else:
        print('There is already a path named Folds, saving folds there')
    
    for fold in range(k_folds):
               
        #Define indices for train and test for the fold
        test_indices = indices[fold*fold_size : (fold+1)*fold_size]
        train_indices = indices[:fold*fold_size] + indices[(fold+1)*fold_size:]
        
        #Extract subsets for training and testing
        test_df = df.iloc[test_indices]
        train_df = df.iloc[train_indices]
        
        #Save folds to csv
        name_test = str(int(fold)) + '_test.csv'
        test_path = os.path.join(directory_name, name_test)
        test_df.to_csv(test_path, index = False)
        name_train = str(int(fold)) + '_train.csv'
        train_path = os.path.join(directory_name, name_train)
        train_df.to_csv(train_path, index = False)
        
path = '/Users/korotkevich/Desktop/Experiments paper/code repo/data/Dm/dm_f_rho.csv' #path to data you want to split
k_folds = 10

create_folds(path, k_folds)       