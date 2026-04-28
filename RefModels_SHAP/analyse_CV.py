#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:52:08 2024

@author: korotkevich
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import argparse

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

def calc_cv_errors(df_dict, col_truth = 'Fitness', col_pred = 'pred'):
    
    #Calculate error for each of the folds
    fold_mae = []
    fold_rmse = []
    fold_r2 = []
    
    for key, df in df_dict.items():
        
        df['a_error'] = abs(df[col_truth] - df[col_pred ])
        df['s_error'] = (df[col_truth] - df[col_pred ])**2
        
        mae = df['a_error'].mean()
        rmse = (df['s_error'].mean())**(0.5)
        r2 = r2_score(df[col_truth], df[col_pred ])

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

        
        print(key)
        # print(f'MAE = {mae}, RMSE = {rmse}')
        # print(f'MAE = {mae}, RMSE = {rmse}')
        
    cv_mae = sum(fold_mae)/len(fold_mae)
    cv_mae_variance = sum([((x - cv_mae) ** 2) for x in fold_mae]) / len(fold_mae) 
    cv_mae_std = cv_mae_variance ** 0.5
    
    cv_rmse = sum(fold_rmse)/len(fold_rmse)
    cv_rmse_variance = sum([((x - cv_rmse) ** 2) for x in fold_rmse]) / len(fold_rmse) 
    cv_rmse_std = cv_rmse_variance ** 0.5

    cv_r2 = np.mean(fold_r2)
    cv_r2_std = np.std(fold_r2)
    
    cv_errors = {'fold_mae':fold_mae, 'fold_rmse':fold_rmse,'fold_r2':fold_r2,\
                 'cv_mae':cv_mae, 'cv_rmse':cv_rmse, 'cv_r2':cv_r2, \
                 'cv_mae_std':cv_mae_std, 'cv_rmse_std':cv_rmse_std, 'cv_r2_std':cv_r2_std 
                 }
    
    return cv_errors
    


def get_CV_stats(dir_path, target_col_name, pred_col_name, fold_vals):
    dfs = csvs_from_dir(dir_path)

    cv_errors = calc_cv_errors(
        dfs,
        col_truth=target_col_name,
        col_pred=pred_col_name
    )

    print('CV MAE', cv_errors['cv_mae'], '+-', cv_errors['cv_mae_std'])
    print('CV RMSE', cv_errors['cv_rmse'], '+-', cv_errors['cv_rmse_std'])
    print('CV R2', cv_errors['cv_r2'], '+-', cv_errors['cv_r2_std'])

    if fold_vals:
    
        print('Fold MAE', cv_errors['fold_mae'])
        print('Fold RMSE', cv_errors['fold_rmse'])
        print('Fold R2', cv_errors['fold_r2'])



def main():
    parser = argparse.ArgumentParser(
        description="Compute CV statistics from saved prediction CSV files.",
        usage=(
            "python analyse_cv.py "
            "--dir RESULTS_DIR "
            "--target-col TARGET_COL "
            "--pred-col PRED_COL"
            "--fold_vals FOLD_VALS"
        )
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing the CV prediction CSV files."
    )

    parser.add_argument(
        "--target-col",
        required=True,
        help="Name of the column containing the true target values."
    )

    parser.add_argument(
        "--pred-col",
        required=True,
        help="Name of the column containing predicted values."
    )

    parser.add_argument(
        "--fold-vals",
        required=False,
        default=False,
        help="Print individual fold values or not. Default False."
    )

    args = parser.parse_args()

    get_CV_stats(
        dir_path=args.dir,
        target_col_name=args.target_col,
        pred_col_name=args.pred_col,
        fold_vals=args.fold_vals
    )


if __name__ == "__main__":
    main()




# plt.figure(figsize = (7,6))
# colors = {1:'g', 2:'r'}
# colors = {1:'g', 2:'g'}
# for key in dfs.keys():
    
    # df = dfs[key]
    # print(df.info())
    #plot all the folds
    # plt.scatter(df['target'], df['pred_0'], c = df.origin.map(colors), s = 1)
#Plot fot 0th fold
# key = 'results/test_fold_4.csv'
# df = dfs[key]
# plt.scatter(df['Fitness'], df['pred'], s = 5)    
# # plt.xlim([0,6])
# # plt.ylim([0,6])
# x = [0,120]
# plt.plot(x,x, color = 'k', linestyle = '--', linewidth = 1)
# plt.xlabel(r'Target', fontsize = 20)
# plt.ylabel(r'Prediction', fontsize = 20)
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 18)
# plt.tight_layout()
# plt.show()