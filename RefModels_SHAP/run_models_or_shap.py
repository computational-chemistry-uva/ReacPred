import json
import os
import argparse

import pandas as pd
import numpy as np


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import shap
from matplotlib import pyplot as plt


def featurize_df(df, fea_dict):
    
    columns = ['Base salt', 'Initial loading', 'Final loading', 'Fitness']
    df = df[columns]
    fea_len = len(list(fea_dict.values())[0]) + 2
    X = np.zeros((len(df), fea_len))                
    for i, row in enumerate(df.itertuples(index = False)):
        salt, il, fl = row[0], float(row[1]), float(row[2])
        fea = fea_dict[salt]
        X[i, :] = np.array(fea + [il,fl])
    
    Y = df['Fitness'].to_numpy()
    
    return X, Y, df



def run_model(path_train,
              path_test,
              fea_dict,
              model_name,
              run_pred=True,
              name='test_result.csv'):
    
    #Load training and testing data
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    X_train, Y_train, _ = featurize_df(df_train, fea_dict)
    X_test, Y_test, df_test = featurize_df(df_test, fea_dict)

    #Select a model
    if model_name == 'RFMagpie':
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    
    elif model_name == 'SVMMagpie':
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.1))
        ])
    
    elif model_name == 'RidgeMagpie':
        
        model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1))
        ])
    else:
        print('model_name can only be "RFMagpie", "SVMMagpie" or "RidgeMagpie"')
        return 1

    #Fit, predict
    model.fit(X_train, Y_train)
        
    # Predict for test set
    Y_pred = model.predict(X_test)
    
    #Get performance stats
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = root_mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    df_test['pred'] = Y_pred
    os.makedirs('results', exist_ok=True)
    path = os.path.join('results', name)
    df_test.to_csv(path, index = False)

    print(f'MAE = {mae}')
    print(f'RMSE = {rmse}')
    print(f'R2 = {r2}')


def run_CV(path_folds, fea_dict, model_name):

    files = os.listdir(path_folds)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    files.sort()

    if model_name is None:
        print('ERROR: no model_name specified')

    i = 0
    while i < len(files):
        
        print(f'FOLD {i//2}')
        
        path_test = os.path.join(path_folds, files[i])
        path_train = os.path.join(path_folds, files[i+1])

        run_model(path_train, path_test, fea_dict, model_name, name=f'test_fold_{i//2}.csv')

        i += 2

def run_feature_analysis(path_train,
                         path_test,
                         fea_dict):
    
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    X_train, Y_train, _ = featurize_df(df_train, fea_dict)
    X_test, _, df_test = featurize_df(df_test, fea_dict)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, Y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=FEATURE_NAMES,
        max_display=10
    )


FEATURE_NAMES = [
    "min_Number", "max_Number", "range_Number", "mean_Number", "avg_dev_Number", "mode_Number",
    "min_MendeleevNumber", "max_MendeleevNumber", "range_MendeleevNumber", "mean_MendeleevNumber", "avg_dev_MendeleevNumber", "mode_MendeleevNumber",
    "min_AtomicWeight", "max_AtomicWeight", "range_AtomicWeight", "mean_AtomicWeight", "avg_dev_AtomicWeight", "mode_AtomicWeight",
    "min_MeltingT", "max_MeltingT", "range_MeltingT", "mean_MeltingT", "avg_dev_MeltingT", "mode_MeltingT",
    "min_Column", "max_Column", "range_Column", "mean_Column", "avg_dev_Column", "mode_Column",
    "min_Row", "max_Row", "range_Row", "mean_Row", "avg_dev_Row", "mode_Row",
    "min_CovalentRadius", "max_CovalentRadius", "range_CovalentRadius", "mean_CovalentRadius", "avg_dev_CovalentRadius", "mode_CovalentRadius",
    "min_Electronegativity", "max_Electronegativity", "range_Electronegativity", "mean_Electronegativity", "avg_dev_Electronegativity", "mode_Electronegativity",
    "min_NsValence", "max_NsValence", "range_NsValence", "mean_NsValence", "avg_dev_NsValence", "mode_NsValence",
    "min_NpValence", "max_NpValence", "range_NpValence", "mean_NpValence", "avg_dev_NpValence", "mode_NpValence",
    "min_NdValence", "max_NdValence", "range_NdValence", "mean_NdValence", "avg_dev_NdValence", "mode_NdValence",
    "min_NfValence", "max_NfValence", "range_NfValence", "mean_NfValence", "avg_dev_NfValence", "mode_NfValence",
    "min_NValence", "max_NValence", "range_NValence", "mean_NValence", "avg_dev_NValence", "mode_NValence",
    "min_NsUnfilled", "max_NsUnfilled", "range_NsUnfilled", "mean_NsUnfilled", "avg_dev_NsUnfilled", "mode_NsUnfilled",
    "min_NpUnfilled", "max_NpUnfilled", "range_NpUnfilled", "mean_NpUnfilled", "avg_dev_NpUnfilled", "mode_NpUnfilled",
    "min_NdUnfilled", "max_NdUnfilled", "range_NdUnfilled", "mean_NdUnfilled", "avg_dev_NdUnfilled", "mode_NdUnfilled",
    "min_NfUnfilled", "max_NfUnfilled", "range_NfUnfilled", "mean_NfUnfilled", "avg_dev_NfUnfilled", "mode_NfUnfilled",
    "min_NUnfilled", "max_NUnfilled", "range_NUnfilled", "mean_NUnfilled", "avg_dev_NUnfilled", "mode_NUnfilled",
    "min_GSvolume_pa", "max_GSvolume_pa", "range_GSvolume_pa", "mean_GSvolume_pa", "avg_dev_GSvolume_pa", "mode_GSvolume_pa",
    "min_GSbandgap", "max_GSbandgap", "range_GSbandgap", "mean_GSbandgap", "avg_dev_GSbandgap", "mode_GSbandgap",
    "min_GSmagmom", "max_GSmagmom", "range_GSmagmom", "mean_GSmagmom", "avg_dev_GSmagmom", "mode_GSmagmom",
    "min_SpaceGroupNumber", "max_SpaceGroupNumber", "range_SpaceGroupNumber", "mean_SpaceGroupNumber", "avg_dev_SpaceGroupNumber", "mode_SpaceGroupNumber",
    "initial_loading", 'final_loading'
]


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-validation or SHAP feature analysis. " \
        "For cross-validation, the results are saved in files in the result directory. " \
        "Use 'analyse_CV' for cross-validation statistics. "
        "SHAP analysis currently supports only Random Forest",
        usage=(
            "python run_models_or_shap.py {cv,feature_analysis} "
            "--features FEATURES_JSON "
            "[--folds FOLDS_DIR] "
            "[--model_name ML_MODEL_ARCH]"
            "[--train TRAIN_CSV --test TEST_CSV]"
        )
    )

    parser.add_argument(
        "mode",
        choices=["cv", "feature_analysis"],
        help="Choose 'cv' for cross-validation or 'feature_analysis' for SHAP analysis."
    )

    parser.add_argument(
        "--features",
        required=True,
        help="Path to the JSON file containing composition feature vectors, e.g. features/magpie_features.json."
    )

    parser.add_argument(
        "--folds",
        help="Path to the folder containing CV train/test fold CSV files. Required when mode='cv'."
    )

    parser.add_argument(
    "--model_name",
        choices=["RFMagpie", "SVMMagpie", "RidgeMagpie"],
        default=None,
        help="Model to use: 'RFMagpie','SVMMagpie', 'RidgeMagpie'"
)

    parser.add_argument(
        "--train",
        help="Path to the training CSV file. Required when mode='feature_analysis'."
    )

    parser.add_argument(
        "--test",
        help="Path to the test CSV file. Required when mode='feature_analysis'."
    )

    args = parser.parse_args()

    with open(args.features, "r") as f:
        fea_dict = json.load(f)

    if args.mode == "cv":
        if args.folds is None:
            parser.error("--folds is required when mode is 'cv'")

        if args.model_name is None:
            parser.error("--model_name is required when mode is 'cv'")

        run_CV(args.folds, fea_dict, args.model_name)

    elif args.mode == "feature_analysis":
        if args.train is None or args.test is None:
            parser.error("--train and --test are required when mode is 'feature_analysis'")

        run_feature_analysis(args.train, args.test, fea_dict)


if __name__ == "__main__":
    main()