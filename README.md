Data
- Data files
The data is stored in the folder Data; there are folders D1, D2, Dm that correspond to the DFT generated data, experimental data and the merged dataset respectively in accordance with the definitions used in the article. Each of these folders contains three .csv files that corresponds three different fitnesses, e.g.:

d1_f_Ev.csv - fitness is volumetric energy density 
d1_f_H.csv - fitness is hydration enthalpy
d1_f_rho.csv - fitness is density

The filenames start with d2 and with dm for D2 and Dm datasets respectively;
In addition, D1 and Dm folders contain folder called ‘Folds’; this folder contains train and test folds used for 10-fold cross-validation for each of the fitnesses;  

If needed, an alternative split can be generated using the script create_folds.py and one of the files with the data that needs to be split.

- Data format
If it is needed to create a custom dataset, the .csv file must have a particular format
Column 1: name - ‘idx’, values - integers; represents number of a data point
Column 2: name - ‘Base salt’, values - strings; represents salt composition; 
Column 3: name - ‘Initial loading’, values - float; represents amount of water in the initial hydrate
Column 4: name - ‘Final loading’, values - float; represents amount of water molecules in the product hydrate
Column 5: name - ‘Fitness’, values - float; represents the target value of the predicted property
Column 6: name - ‘Origin’, values - optional; represents origin of the data; if this column is not present a dummy column will be created; else, the provided origins will be replaced by numeric values, the replacement rule will be printed


Remarks
1) If a dataset is going to be used only for predictions with a retrained model and the target value is unknown - the ’Fitness’ column must still be present and contain some dummy numerical values (e.g. zeros)
2) Despite the names ‘Base salt’, ‘Initial loading’, ‘Final loading’ are related to salt hydration reaction, the models can be applied to any addition reaction of type S.mL + (n-m)L -> S.nL where S is a composition of a substrate, n and m are stoichiometric amounts of a certain ligand L (identical for the dataset)

Embeddings
The folder embeddings contains element embeddings as reported in the original Roost paper 
(Goodall, R.E.A., Lee, A.A,  Nat Commun 11, 6280 (2020))
Additionally, it contains Magpie features for data used for model training on Dm data (magpie_features.json);  this file can also be used to run tasks on the D1 data only 
If MagPieNN (ReacCryNN) predictions/training is desired for some compositions that are not in that data, new Magpie features must be generated before running the model using the script create_magpie.py

(Quick) Setup
1) Setup a Python environment. We recommend using Python 3.9.19, as this version has been used throughout the project 
If you use conda, create a new environment, and activate it in Terminal:
conda create -n env_name python=3.9.19
conda activate env_name

2) Instal libraries - the libraries and their versions are saved in the requirements.txt.
In conda, after creating and activating the environment, install requirements using pip
pip install -r requirements.txt

3) In your IDE open the file workspace.py, and run the code. If everything is set up right, this should start a single train/test split run using ReacRoostC on the d1 dataset, for getting the model predicting densities. 

4) The prediction results will be saved in the folder results, and the model will be saved in the folder model


Running tasks
Arguments
In the file workspace.py there is a function called input_parser() which is generating the arguments for task initialisation with default values
args = input_parser()
This can be used to run tasks from Terminal, as designed in the original paper

We use an alternative approach, namely running tasks from an IDE
Firstly, args are converted into a Python dict by:
args = vars(args)
This dict can be printed if it’s necessary to see the argument keywords and default values, alternatively the input_parcer() can be examined;

The args can be modified with custom values using arguments’ names as dict keys
Some commonly modified arguments are:

args['train'] = bool - specifies whether the model has to be trained or not; if it’s False, a checkpoint with a prerained model is required

args[‘evaluate’] = bool - specifies whether a model has to be evaluated on an independent data set and save results into the Results directory; the path to the test set can be provided (see input_parser); if not evaluates data on the validation set

args['fine_tune'] = bool - specifies path for the checkpoint of a pretrained model; checkpoints are saved in the models directory; it looks like /../models/roost_10_epochs/checkpoint-r0.pth.tar’; NOTE, if train is True, the model will be retrained, so if only evaluation on a test set is needed is needed, use args['train'] = False

args['data_path'] = str - specifies path to data, by uses 0.8 of the data for training and 0.2 for validation/test if no test_path is provided

args[‘test_path’] = str - specifies path to the test set; if provided and validation set is not specified uses the test set for validation during the training; if not provided by default uses 0.2 of the data in the data_path as a validation and test set

args[‘epoch’] =  int - specifies number of epochs, i.e. full passes through training data

args['model_class'] = str - specifies model type, can be only one of ReacCryNet, ReacElemNet, ReacRoost
args['fea_path'] = str - specifies a path to the features/embeddings; if ReacCryNet is used as model class - requires a path to crystal features; if ReacElemNet or ReacRoost is used - requires path to element embeddings

args['elem_fea_len'] = int - specifies dimension of the feature/element embedding vector; if dimension reduction is not used (see below), must be the same as the original feature/embedding length;
args[‘append_after] = str - specifies whether loadings are appended to the element embeddings before (‘E’) or after (‘C’) message passing operations. Makes a difference only for model type ReacRoost, for others it’s always ‘C’;

args['dim_red'] = bool - specifies, if learnable dimension reduction is performed on features/embeddings before message passing and/or predictive model;
if True elem_fea_len must be < elem_emb_len, 
if True, in case model_class is ReacCryNet, elem_fea_len corresponds to the embedding length BEFORE appending loadings, i.e. the dimension of feature used as input of final predictive model is elem_fea_len + 2;
if True, in case model_class is ReacElemNet, elem_fea_len corresponds to embedding length BEFORE appending loadings, i.e. the dimension of feature used as input of final predictive model is elem_fea_len + 2;
if True, in case model_class is ReacRoost AND append_after = ‘C’ elem_fea_len corresponds to embedding length BEFORE appending the loadings, but AFTER appending the atomic fraction (weight), i.e. the dimension of feature after the dimension reduction step is elem_fea_len -1, the dimension of feature after appending the atomic fraction (weight) and in the message passing section is elem_fea_len, and eventually the dimension of feature used as input of final predictive model is elem_fea_len + 2
if True, in case model_class is ReacRoost AND append_after = ‘E’ elem_fea_len corresponds to embedding length AFTER appending  BOTH the loadings the atomic fractions (weights), i.e. the dimension of feature after the dimension reduction step is elem_fea_len - 3, the dimension of feature after appending both the atomic fraction (weight) and the loadings and consequently in the message passing section is elem_fea_len, and eventually the dimension of feature used as input of final predictive model is elem_fea_len
if False, in case model_class is ReacRoost AND append_after = ‘C’ elem_fea_len corresponds to the original embedding length, i.e. the dimension of feature used as input of final predictive model is elem_fea_len + 3 (two loadings + one atomic fraction (weight))
if False, in case model_class is ReacRoost AND append_after = ‘E’ elem_fea_len corresponds to the original embedding length, i.e. the dimension of feature used as input of final predictive model is elem_fea_len + 3 (two loadings + one atomic fraction (weight))

Running
The tasks are run by calling main(**args) in the workspace.py.
A single train/test split run or an evaluation of a preretrained on an independent test set is done by a single function call (see #Running a task on a single train/test split in workspace.py). The test results are saved in the Results directory.
A k-fold cross-validation task is run in a loop of k iterations (see #Running a cross-validation task in workspace.py). Before running, k train test splits of data must be created, which can be done using script create_folds.py. The model parameters and the path to the directory with the splits (folds) are specified before looping. In the loop, the train/test splits and model names are alternated at each iteration and the results are saved in the Results folder. The results can be analysed using the analyse_CV.py script. NOTE the Results are not necessarily overwritten, so before analysing cross-validation results, make sure, the folder contains only the necessary files with the results.

Remarks
Make sure to comment/uncomment unnecessary parts: the workspace can run either a single train/test split task or a k-fold cross-validation. A prediction with a retrained model without training can be set up in the a single train/test split, by disabling training (args['train'] = False), providing paths to a checkpoint with a pre-trained model (args['fine_tune'] = ‘’path’) and to an independent test set (args[‘test_path’] = path)
