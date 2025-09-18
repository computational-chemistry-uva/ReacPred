#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:52:08 2024

@author: korotkevich
"""

import pandas as pd
import os
from matplotlib import pyplot as plt

def csvs_from_dir(dir_path, start = 'test'):
    '''
    This function reads a number of .csv files into dfs and returns
    them as a dict, where keys are paths and values are dfs. Uses the key
    start to identify the proper file names

    Parameters
    ----------
    dir_path : str
        path to the directory with results
    start : str, optional
         start of the name of the file; the default is 'test'.

    Returns
    -------
    dict with pandas dfs

    '''
    
    dfs_cv = {}
    for f in os.listdir(dir_path):
       
        f_path = os.path.join(dir_path, f)
        
        #Check if it is a file
        if not os.path.isfile(f_path):
            continue
            
        
        #Check if the name starts as we need     
        elif not f.startswith(start):
            continue
        
        else:
            
            dfs_cv[f_path] = pd.read_csv(f_path)
            
        
    return dfs_cv

def calc_cv_errors(df_dict):
    
    #Calculate error for each of the folds
    fold_mae = []
    fold_rmse = []
    
    for key, df in dfs.items():
        
        df['a_error'] = abs(df['target'] - df['pred_0'])
        df['s_error'] = (df['target'] - df['pred_0'])**2
        
        mae = df['a_error'].mean()
        rmse = (df['s_error'].mean())**(0.5)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        
        # print(key)
        # print(f'MAE = {mae}, RMSE = {rmse}')
        
    cv_mae = sum(fold_mae)/len(fold_mae)
    cv_mae_variance = sum([((x - cv_mae) ** 2) for x in fold_mae]) / len(fold_mae) 
    cv_mae_std = cv_mae_variance ** 0.5
    
    cv_rmse = sum(fold_rmse)/len(fold_rmse)
    cv_rmse_variance = sum([((x - cv_rmse) ** 2) for x in fold_rmse]) / len(fold_rmse) 
    cv_rmse_std = cv_rmse_variance ** 0.5
    
    cv_errors = {'fold_mae':fold_mae, 'fold_rmse':fold_rmse,\
                 'cv_mae':cv_mae, 'cv_rmse':cv_rmse,\
                 'cv_mae_std':cv_mae_std, 'cv_rmse_std':cv_rmse_std,}
    
    return cv_errors
    
       
dir_path = 'results'
dfs = csvs_from_dir(dir_path)
cv_errors = calc_cv_errors(dfs)
print('CV MAE', cv_errors['cv_mae'], '+-', cv_errors['cv_mae_std'])
print('CV RMSE', cv_errors['cv_rmse'], '+-', cv_errors['cv_rmse_std'])


plt.figure(figsize = (7,6))
# colors = {1:'g', 2:'r'}
colors = {1:'g', 2:'g'}
for key in dfs.keys():
    
    df = dfs[key]
    # print(df.info())
    #plot all the folds
    # plt.scatter(df['target'], df['pred_0'], c = df.origin.map(colors), s = 1)
#Plot fot oth fold
key = 'results/test_results_roost_CV_fold_0_r-0.csv'
df = dfs[key]
plt.scatter(df['target'], df['pred_0'], c = df.origin.map(colors), s = 5)    
plt.xlim([0,120])
plt.ylim([0,120])
x = [0,120]
plt.plot(x,x, color = 'k', linestyle = '--', linewidth = 1)
plt.xlabel(r'$\Delta H_{target}, kJ/mol$', fontsize = 20)
plt.ylabel(r'$\Delta H_{prediction}, kJ/mol$', fontsize = 20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)