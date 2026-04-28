import pandas as pd
import numpy as np
import json
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import argparse


class MahalanobisEmpirical:
    def __init__(self, X_train, regularize=1e-8):
        """
        X_train: (n, d) training data
        regularize: small diagonal term for numerical stability
        """
        self.X_train = X_train
        self.n, self.d = X_train.shape
        
        # Mean
        self.mu = X_train.mean(axis=0)
        
        # Covariance with small regularization
        cov = np.cov(X_train, rowvar=False)
        cov += regularize * np.eye(self.d)
        
        # Inverse covariance
        self.inv_cov = np.linalg.inv(cov)
        
        # Precompute training squared distances
        diffs = X_train - self.mu
        self.train_D2 = np.einsum('ij,jk,ik->i', diffs, self.inv_cov, diffs)

        self.min_train_D2 = self.train_D2.min()
        self.max_train_D2 = self.train_D2.max()
    
    def distance_from_mean(self, x):
        diff = x - self.mu
        return diff @ self.inv_cov @ diff
    
    def distances_from_training(self, x):
        """
        Returns squared Mahalanobis distances from x to every training sample.
        Output shape: (n,)
        """
        diffs = self.X_train - x   # shape (n, d)
        return np.einsum('ij,jk,ik->i', diffs, self.inv_cov, diffs)
    
    
    def percentile(self, x):
        """
        Returns empirical percentile of x relative to training distances.
        """
        d2 = self.distance_from_mean(x)
        return np.mean(self.train_D2 <= d2)
    
    def plot_distribution_with_point(self, xnew_list, bins=100, density=False, show=True):
        """
        Plot histogram of training squared Mahalanobis distances.
        """

        colors = ['r', 'b', 'g']
        salts = ['Rb2Cu(SO4)2', 'Rb2Fe(SO4)2', 'Rb2Zn(SO4)2']
        salt_legs = [r"$\mathrm{Rb_2Cu(SO_4)_2}$", r"$\mathrm{Rb_2Fe(SO_4)_2}$", r"$\mathrm{Rb_2Zn(SO_4)_2}$"]
        plt.figure(figsize=(8,8))
        plt.hist(self.train_D2, bins=bins, density=density, alpha=0.7, edgecolor="black")
        d2_list = []

        p10 = np.percentile(self.train_D2, 10)
        p30 = np.percentile(self.train_D2, 30)
        plt.axvline(p10, color="gray", linewidth=2, label="10th percentile")
        plt.axvline(p30, color="k", linewidth=2, label="30th percentile")
        for x_new, salt_leg, c in zip(xnew_list, salt_legs, colors):
            d2 = self.distance_from_mean(x_new)
            d2_list.append(d2)
            plt.axvline(d2, linestyle="--", color = c,  linewidth=2, label=f"{salt_leg}")
        


        plt.xlabel("Squared Mahalanobis distance")
        plt.ylabel("Density" if density else "Count")
        # plt.title("Distribution of training squared Mahalanobis distances")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim([0,500])
        if show:
            plt.show()

def explore_mahalanobis(fea_type):
    
    path_salts = 'unique_salts_dm.csv'
    df = pd.read_csv(path_salts)

    if fea_type == 'magpie':
        
        path_fea = 'embeddings/magpie_features.json'
        path_fea_new = 'embeddings/magpie_features_input.json'

    elif fea_type == 'onehot':
        
        path_fea = 'embeddings/wp_onehot_salts.json'
        path_fea_new = 'embeddings/wp_onehot_test_salts.json'

    elif fea_type == 'roost_Ev':

        path_fea = 'embeddings/emb_roost_Ev_train_model1.json'
        path_fea_new = 'embeddings/emb_roost_Ev_test_model1.json'
    
    else:
        print('ERROR. fea_type can only be "magpie", "onehot" or "roost_Ev"')


    with open(path_fea, 'r') as f:   
        reps_dict = json.load(f)


    with open(path_fea_new, 'r') as f:   
        reps_dict_new = json.load(f)

    keys = []
    feas = []
    for key, fea in reps_dict.items():
        
        keys.append(key)
        feas.append(np.array(fea))

    feas = np.array(feas)

    model = MahalanobisEmpirical(feas)
    print(f'Min distance between the vectors in the training data = {round(model.min_train_D2, 3)}')
    print(f'Max distance between the vectors in the training data = {round(model.max_train_D2, 3)}')


    for salt in ['Rb2Cu(SO4)2', 'Rb2Fe(SO4)2', 'Rb2Zn(SO4)2']:
        x_new = np.array(reps_dict_new[salt])
        d2_new = round(model.distance_from_mean(x_new), 3)
        percentile = round(model.percentile(x_new), 3)
        print(f"Representation of {salt} has Mahalanobis distance of {d2_new} from the mean vector")
        print(f"Representation of {salt} is farther from the space mean than {percentile} of the vectors")

    x_list = []
    for salt in ['Rb2Cu(SO4)2', 'Rb2Fe(SO4)2', 'Rb2Zn(SO4)2']:
        x_new = np.array(reps_dict_new[salt])
        x_list.append(x_new)

    model.plot_distribution_with_point(x_list)



def main():
    parser = argparse.ArgumentParser(
    description="Explore Mahalanobis distances for salt feature representations.",
    usage="python explore_mahalanobis.py {magpie,onehot,roost_Ev}"
    )

    parser.add_argument(
        "fea_type",
        choices=["magpie", "onehot", "roost_Ev"],
        help="Feature type to use."
    )

    args = parser.parse_args()
    explore_mahalanobis(args.fea_type)


if __name__ == "__main__":
    main()
