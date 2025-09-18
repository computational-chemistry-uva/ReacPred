import functools
import os
import re
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core import LoadFeaturiser

#SEARCH FOR THESE CODES TO SEE WHAT WAS MODIFIED IN COMPARISON TO THE ORIGINAL CODE
#MPHE = #MODIFICATION FOR PREDICTION OF HYDRATION ENTHALPY
#Orig - original code

class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """
#MPHE Added variable append_after
    def __init__(self, data_path, fea_path, task, append_after):
        """
        """
        assert os.path.exists(data_path), "{} does not exist!".format(data_path)
        # NOTE make sure to use dense datasets, here do not use the default na
        # as they can clash with "NaN" which is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])
        print('Columns found in the data set')
        for column in list(self.df.columns):
            print(column)
        #Check if Origin column is present, if not not create one filled with 0
        if len(self.df.columns) < 6 or 'Origin' not in self.df.columns:
            print('No Origin column found, creating a dummy column')
            self.df['Origin'] = 0
        #Convert origins into numeric format
        else:
            origins = {}
            for i, origin in enumerate(self.df['Origin'].unique()):
               origins[origin] = i
            print("Replacing the provided values in 'Origin' by numeric values as:")
            for key, value in origins.items():
                print(f'{key} -> {value}')
            pd.set_option('future.no_silent_downcasting', True)
            self.df['Origin'] = self.df['Origin'].replace(origins)
        print('END OF DATA PROCESSING')
        
        # assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.elem_features = LoadFeaturiser(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size
        self.task = task
        
        self.append_after = append_after
        
        if self.task == "regression":
        #MPHE
            # if self.df.shape[1] - 2 != 1:
            #     raise NotImplementedError(
            #         "Multi-target regression currently not supported"
            #     )
            self.n_targets = 1
        elif self.task == "classification":
            if self.df.shape[1] - 2 != 1:
                raise NotImplementedError(
                    "One-Hot input not supported please use categorical integer"
                    " inputs for classification i.e. Dog = 0, Cat = 1, Mouse = 2"
                )
            self.n_targets = np.max(self.df[self.df.columns[2]].values) + 1

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        #__getitem__ allows an instance of a class to behave as a collection
        #like list, tuple etc.; i.e. one can call self[idx] and get the 
        #output of __getitem__. Here it will be a set of model input arguments
        #that corresponds to a single database entry
        """

        Returns
        -------
        atom_weights: torch.Tensor shape (M, 1)
            weights of atoms in the material
        atom_fea: torch.Tensor shape (M, n_fea)
            features of atoms in the material
        self_fea_idx: torch.Tensor shape (M*M, 1)
            list of self indices
        nbr_fea_idx: torch.Tensor shape (M*M, 1)
            list of neighbor indices
        target: torch.Tensor shape (1,)
            target value for material
        cry_id: torch.Tensor shape (1,)
            input id for the material
        """
        #Orig
        #cry_id, composition, target = self.df.iloc[idx]
        
        #=============================================================================
        #MPHE
        cry_id, composition, init_num, fin_num, target, origin = self.df.iloc[idx]
        hydration_states = [float(init_num), float(fin_num)]
        # if origin == 's':
        #     origin = 1
        # elif origin =='e':
        #     origin = 2

        #=============================================================================

        elements, weights = parse_roost(composition)
        if self.append_after == "C":
            hydration_states_input = hydration_states
        elif self.append_after == "E":
            hydration_states_input = np.array([hydration_states for i in range(len(weights))])
        weights = np.atleast_2d(weights).T / np.sum(weights)

        assert len(elements) != 1, f"cry-id {cry_id} [{composition}] is a pure system"
        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_id} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_id} [{composition}] composition cannot be parsed into elements"
            )

        env_idx = list(range(len(elements)))
        self_fea_idx = []
        nbr_fea_idx = []
        nbrs = len(elements) - 1
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nbrs
            nbr_fea_idx += env_idx[:i] + env_idx[i + 1 :]

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        #MPHE
        hydration_states_input = torch.tensor(hydration_states_input)
        hydration_states_results = hydration_states
        origin = torch.tensor(origin)
        if self.task == "regression":
            targets = torch.Tensor([float(target)])
        elif self.task == "classification":
            targets = torch.LongTensor([target])

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx,\
             #MPHE
             hydration_states_input), #this one will be used for model input
            targets,
            composition,
            #MPHE
            hydration_states_results,#this one will be used for writing the results
            origin,#this is used to label whether a datapoint comes from d1 or d2
            cry_id,
            self.append_after
        )


def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_target = []
    batch_comp = []
    #MPHE
    batch_hydration_states = []
    batch_hydration_states_results = []
    batch_origins = []
    batch_cry_ids = []

    cry_base_idx = 0
    #Orig
    # for i, (inputs, target, comp, cry_id) in enumerate(dataset_list):
    #   atom_weights, atom_fea, self_fea_idx, nbr_fea_idx = inputs

    #MPHE
# =============================================================================
    for i, (inputs, target, comp, hydration_states_results, origin, cry_id, append_after) in enumerate(dataset_list):    
        
        atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, hydration_states_input = inputs
# =============================================================================
        
        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)
        #MPHE
        batch_hydration_states.append(hydration_states_input)
    
        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_target.append(target)
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)
        batch_origins.append(origin)
        batch_hydration_states_results.append(hydration_states_results)


        # increment the id counter
        cry_base_idx += n_i
        
    if append_after == "E":
        batch_hydration_states = torch.cat(batch_hydration_states, dim=0)
    elif append_after == "C":
        batch_hydration_states = torch.stack(batch_hydration_states, dim=0)
        
    return (
        (
            torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx),
            #MPHE
            batch_hydration_states
            
        ),
        torch.stack(batch_target, dim=0),
        batch_comp,
        batch_hydration_states_results,
        batch_origins,
        batch_cry_ids,
    )


def format_composition(comp):
    """ format str to ensure weights are explicate
    example: BaCu3 -> Ba1Cu3
    """
    subst = r"\g<1>1.0"
    comp = re.sub(r"[\d.]+", lambda x: str(float(x.group())), comp.rstrip())
    comp = re.sub(r"([A-Z][a-z](?![0-9]))", subst, comp)
    comp = re.sub(r"([A-Z](?![0-9]|[a-z]))", subst, comp)
    comp = re.sub(r"([\)](?=[A-Z]))", subst, comp)
    comp = re.sub(r"([\)](?=\())", subst, comp)
    return comp


def parenthetic_contents(string):
    """
    Generate parenthesized contents in string as (level, contents, weight).
    """
    num_after_bracket = r"[^0-9.]"

    stack = []
    for i, c in enumerate(string):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            start = stack.pop()
            num = re.split(num_after_bracket, string[i + 1 :])[0] or 1
            yield {
                "value": [string[start + 1 : i], float(num), False],
                "level": len(stack) + 1,
            }

    yield {"value": [string, 1, False], "level": 0}


def splitout_weights(comp):
    """ split composition string into elements (keys) and weights
    example: Ba1Cu3 -> [Ba,Cu] [1,3]
    """
    elements = []
    weights = []
    regex3 = r"(\d+\.\d+)|(\d+)"
    try:
        parsed = [j for j in re.split(regex3, comp) if j]
    except:
        print("parsed:", comp)
    elements += parsed[0::2]
    weights += parsed[1::2]
    weights = [float(w) for w in weights]
    return elements, weights


def update_weights(comp, weight):
    """ split composition string into elements (keys) and weights
    example: Ba1Cu3 -> [Ba,Cu] [1,3]
    """
    regex3 = r"(\d+\.\d+)|(\d+)"
    parsed = [j for j in re.split(regex3, comp) if j]
    elements = parsed[0::2]
    weights = [float(p) * weight for p in parsed[1::2]]
    new_comp = ""
    for m, n in zip(elements, weights):
        new_comp += m + f"{n:.2f}"
    return new_comp


class Node(object):
    """ Node class for tree data structure """

    def __init__(self, parent, val=None):
        self.value = val
        self.parent = parent
        self.children = []

    def __repr__(self):
        return f"<Node {self.value} >"


def build_tree(root, data):
    """ build a tree from ordered levelled data """
    for record in data:
        last = root
        for _ in range(record["level"]):
            last = last.children[-1]
        last.children.append(Node(last, record["value"]))


def print_tree(current, depth=0):
    """ print out the tree structure """
    for child in current.children:
        print("  " * depth + "%r" % child)
        print_tree(child, depth + 1)


def reduce_tree(current):
    """ perform a post-order reduction on the tree """
    if not current:
        pass

    for child in current.children:
        reduce_tree(child)
        update_parent(child)


def update_parent(child):
    """ update the str for parent """
    input_str = child.value[2] or child.value[0]
    new_str = update_weights(input_str, child.value[1])
    pattern = re.escape("(" + child.value[0] + ")" + str(child.value[1]))
    old_str = child.parent.value[2] or child.parent.value[0]
    child.parent.value[2] = re.sub(pattern, new_str, old_str, 0)


def parse_roost(string):
    # format the string to remove edge cases
    string = format_composition(string)
    # get nested bracket structure
    nested_levels = list(parenthetic_contents(string))
    if len(nested_levels) > 1:
        # reverse nested list
        nested_levels = nested_levels[::-1]
        # plant and grow the tree
        root = Node("root", ["None"] * 3)
        build_tree(root, nested_levels)
        # reduce the tree to get compositions
        reduce_tree(root)
        return splitout_weights(root.children[0].value[2])

    else:
        return splitout_weights(string)

class CryCompositionData(Dataset):
    
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings using predefined
    featurization schemes like Magpie
    """

    def __init__(self, data_path, fea_path, task, append_after):
        """
        """
        assert os.path.exists(data_path), "{} does not exist!".format(data_path)
        # NOTE make sure to use dense datasets, here do not use the default na
        # as they can clash with "NaN" which is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])
        print('Columns found in the data set')
        for column in list(self.df.columns):
            print(column)
        
        #Check if Origin column is present, if not create one filled with 0
        if len(self.df.columns) < 6 or 'Origin' not in self.df.columns:
            print('No Origin column found, creating a dummy column')
            self.df['Origin'] = 0
        #Convert origins into numeric format
        else:
            origins = {}
            for i, origin in enumerate(self.df['Origin'].unique()):
               origins[origin] = i
            print("Replacing the provided values in 'Origin' by numeric values as:")
            for key, value in origins.items():
                print(f'{key} -> {value}')
            pd.set_option('future.no_silent_downcasting', True)
            self.df['Origin'] = self.df['Origin'].replace(origins)
        print('END OF DATA PROCESSING')    
        with open(fea_path, 'r') as file:
            self.features = json.load(file)
        
        self.task = task
        self.append_after = append_after
        self.elem_emb_len = len(list(self.features.values())[0])


        if self.task == "regression":
            self.n_targets = 1
        elif self.task == "classification":
            if self.df.shape[1] - 2 != 1:
                raise NotImplementedError(
                    "One-Hot input not supported please use categorical integer"
                    " inputs for classification i.e. Dog = 0, Cat = 1, Mouse = 2"
                )
            self.n_targets = np.max(self.df[self.df.columns[2]].values) + 1

    def __len__(self):
        return len(self.df)
    
    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        #__getitem__ allows an instance of a class to behave as a collection
        #like list, tuple etc.; i.e. one can call self[idx] and get the 
        #output of __getitem__. Here it will be a set of model input arguments
        #that corresponds to a single database entry

        
        cry_id, composition, init_num, fin_num, target, origin = self.df.iloc[idx]
        hydration_states = [float(init_num), float(fin_num)]
        
        # if origin == 's':
        #     origin = 1
        # elif origin =='e':
        #     origin = 2
       

        salt_fea = self.features[composition]
        salt_fea = torch.Tensor(salt_fea)
        hydration_states_input = torch.tensor(hydration_states)
        hydration_states_results = hydration_states
        origin = torch.tensor(origin)
        
        if self.task == "regression":
            targets = torch.Tensor([float(target)])
        elif self.task == "classification":
            targets = torch.LongTensor([target])

        return (
            (salt_fea,
             hydration_states_input), #this one will be used for model input
            targets,
            composition,
            hydration_states_results,
            origin, #this one will be used for writing the results
            cry_id,
        )

def cry_collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties for the case when composition strings are processed using a
    standard featurization scheme like Magpie

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (salt_fea, hydration_states_input)

      salt_fea: torch.Tensor shape (n_i, magpie_fea_len)
      target: torch.Tensor shape (1, )


    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_salt_fea: torch.Tensor shape (N, magpie_fea_len)
        Atom features from atom type
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
    """
    # define the lists
    batch_salt_fea = []
    batch_hydration_states = []
    batch_target = []
    batch_comp = []
    batch_hydration_states_results = []
    batch_cry_ids = []
    batch_origins = []

    # cry_base_idx = 0

    for i, (inputs, target, comp, hydration_states_results, origin, cry_id) in enumerate(dataset_list):    
        
        salt_fea, hydration_states_input = inputs
        # fea = torch.cat((salt_fea, hydration_states_input.float()), dim = 0)
        
        batch_salt_fea.append(salt_fea)

        batch_hydration_states.append(hydration_states_input)
        batch_target.append(target)
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)
        batch_hydration_states_results.append(hydration_states_results)
        batch_origins.append(origin)

    
    return (
        (

            torch.stack(batch_salt_fea, dim=0),
            torch.stack(batch_hydration_states, dim=0),
        ),
        torch.stack(batch_target, dim=0),
        batch_comp,
        batch_hydration_states_results,
        batch_origins,
        batch_cry_ids,
    )