#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:14:33 2024

@author: korotkevich
"""

import pandas as pd
import json

from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty as ep

path_data = '/Users/korotkevich/Desktop/Experiments paper/code repo/data/Dm/dm_f_H.csv'
df = pd.read_csv(path_data)

#Extract unique stoichiometries
compositions = df['Base salt'].unique().tolist() #column with stoichiometries

#Initialize featurizer
featurizer = ep.from_preset("magpie")

magpie_features = {}

for salt in compositions:
    
    comp = Composition(salt)
    magpie_feature = featurizer.featurize(comp)
    magpie_features[salt] = magpie_feature


    
with open('magpie_features_file.json', 'w') as f:
    json.dump(magpie_features, f)


 