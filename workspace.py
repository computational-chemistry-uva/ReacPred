import os
import sys
import argparse

import torch
from sklearn.model_selection import train_test_split as split

from model import Roost
from reac_model import ReacRoost, ReacElemNet, ReacCryNet
from data import (
    CompositionData, 
    CryCompositionData,
    collate_batch, 
    cry_collate_batch
)
from utils import (
    train_ensemble,
    results_regression,
    results_classification,
)


def main(
    data_path,
    fea_path,
    task,
    loss,
    robust,
    model_class = "ReacElemNet",
    model_name="roost",
    elem_fea_len=64,
    n_graph=3,
    ensemble=1,
    run_id=1,
    data_seed=42,
    epochs=100,
    log=True,
    sample=1,
    test_size=0.2,
    test_path=None,
    val_size=0.0,
    val_path=None,
    resume=None,
    fine_tune=None,
    transfer=None,
    train=True,
    evaluate=True,
    optim="AdamW",
    learning_rate=3e-4,
    momentum=0.9,
    weight_decay=1e-6,
    batch_size=128,
    workers=0,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    append_after = "C",
    dim_red = True,
    **kwargs,
):
    assert evaluate or train, (
        "No task given - Set at least one of 'train' or 'evaluate' kwargs as True"
    )
    assert task in ["regression", "classification"], (
        "Only 'regression' or 'classification' allowed for 'task'"
    )

    if test_path:
        test_size = 0.0

    if not (test_path and val_path):
        assert test_size + val_size < 1.0, (
            f"'test_size'({test_size}) "
            f"plus 'val_size'({val_size}) must be less than 1"
        )

    if ensemble > 1 and (fine_tune or transfer):
        raise NotImplementedError(
            "If training an ensemble with fine tuning or transfering"
            " options the models must be trained one by one using the"
            " run-id flag."
        )

    assert not (fine_tune and transfer), (
        "Cannot fine-tune and" " transfer checkpoint(s) at the same time."
    )
    
    if model_class == "ReacElemNet" or model_class == "ReacCryNet":
        assert append_after == "C", (
            f"For model_class = {model_class} can only use append_after = C"
            )

    if model_class == "ReacElemNet" or model_class == "ReacRoost":
        DataWrapper = CompositionData
    elif model_class == "ReacCryNet":
        DataWrapper = CryCompositionData
    
    dataset = DataWrapper(data_path=data_path, fea_path=fea_path, task=task, append_after = append_after)
    n_targets = dataset.n_targets
    elem_emb_len = dataset.elem_emb_len
    
    if not dim_red:
        assert elem_emb_len == elem_fea_len, (
            "When embedding dimension reduction is not used, (dim_red = False)" 
            f" elem_fea_len must be the same as elem_emb_len = {elem_emb_len}."
            " Please check the length of the elem_fea_len used"
            )
    else:
        assert elem_emb_len > elem_fea_len, (
            "When embedding dimension reduction is requested, (dim_red = True)"
            f" elem_fea_len must be less than elem_emb_len = {elem_emb_len}."
            " Please check the length of the elem_fea_len used"
            )
            

    train_idx = list(range(len(dataset)))

    if evaluate:
        if test_path:
            print(f"using independent test set: {test_path}")
            test_set = DataWrapper(
                data_path=test_path, fea_path=fea_path, task=task, append_after = append_after
            )
            test_set = torch.utils.data.Subset(test_set, range(len(test_set)))
        elif test_size == 0.0:
            raise ValueError("test-size must be non-zero to evaluate model")
        else:
            print(f"using {test_size} of training set as test set")
            train_idx, test_idx = split(
                train_idx, random_state=data_seed, test_size=test_size
            )
            test_set = torch.utils.data.Subset(dataset, test_idx)

    if train:
        if val_path:
            print(f"using independent validation set: {val_path}")
            val_set = DataWrapper(data_path=val_path, fea_path=fea_path, task=task, append_after = append_after)
            val_set = torch.utils.data.Subset(val_set, range(len(val_set)))
        else:
            if val_size == 0.0 and evaluate:
                print("No validation set used, using test set for evaluation purposes")
                # NOTE that when using this option care must be taken not to
                # peak at the test-set. The only valid model to use is the one
                # obtained after the final epoch where the epoch count is
                # decided in advance of the experiment.
                val_set = test_set
            elif val_size == 0.0:
                val_set = None
            else:
                print(f"using {val_size} of training set as validation set")
                train_idx, val_idx = split(
                    train_idx, random_state=data_seed, test_size=val_size / (1 - test_size),
                )
                val_set = torch.utils.data.Subset(dataset, val_idx)

        train_set = torch.utils.data.Subset(dataset, train_idx[0::sample])

    data_params = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch

    }
    
    if model_class == "ReacCryNet":
        
        data_params["collate_fn"] = cry_collate_batch

    setup_params = {
        "loss": loss,
        "optim": optim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "device": device,
    }

    restart_params = {
        "resume": resume,
        "fine_tune": fine_tune,
        "transfer": transfer,
    }
    
    
    #Basic set of parameters which is necessary for ReacNN
    model_params = {
        "task": task,
        "robust": robust,
        "n_targets": n_targets,
        "out_hidden": [1024, 512, 256, 128, 64],
        "dim_red": dim_red
    }
    
    if model_class == "ReacRoost":
        model_class = ReacRoost
        
    #Additional set of parameters necessary for ReacRoost
        ReacRoost_params = {"elem_emb_len": elem_emb_len,
                "elem_fea_len": elem_fea_len,
                "n_graph": n_graph,
                "elem_heads": 3,
                "elem_gate": [256],
                "elem_msg": [256],
                "cry_heads": 3,
                "cry_gate": [256],
                "cry_msg": [256],
                "append_after": append_after}
        
        model_params.update(ReacRoost_params)
    
    elif model_class == "ReacElemNet":
        model_class = ReacElemNet
        
        ReacElemNet_params = {"elem_emb_len": elem_emb_len,
                "elem_fea_len": elem_fea_len}
        
        model_params.update(ReacElemNet_params)
    
    elif model_class == "ReacCryNet":       
        model_class = ReacCryNet
        
        ReacCryNet_params = {"cry_emb_len": elem_emb_len,
                "cry_fea_len": elem_fea_len}
        
        model_params.update(ReacCryNet_params)
        
    os.makedirs(f"models/{model_name}/", exist_ok=True)

    if log:
        os.makedirs("runs/", exist_ok=True)

    os.makedirs("results/", exist_ok=True)

    if train:
        train_ensemble(
            model_class=model_class,
            model_name=model_name,
            run_id=run_id,
            ensemble_folds=ensemble,
            epochs=epochs,
            train_set=train_set,
            val_set=val_set,
            log=log,
            data_params=data_params,
            setup_params=setup_params,
            restart_params=restart_params,
            model_params=model_params,
        )

    if evaluate:

        data_reset = {
            "batch_size": 16 * batch_size,  # faster model inference
            "shuffle": False,  # need fixed data order due to ensembling
        }
        data_params.update(data_reset)

        if task == "regression":
            results_regression(
                model_class=model_class,
                model_name=model_name,
                run_id=run_id,
                ensemble_folds=ensemble,
                test_set=test_set,
                data_params=data_params,
                robust=robust,
                device=device,
                eval_type="checkpoint",
            )
        elif task == "classification":
            results_classification(
                model_class=model_class,
                model_name=model_name,
                run_id=run_id,
                ensemble_folds=ensemble,
                test_set=test_set,
                data_params=data_params,
                robust=robust,
                device=device,
                eval_type="checkpoint",
            )


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(
        description=(
            "Roost - a Structure Agnostic Message Passing "
            "Neural Network for Inorganic Materials"
        )
    )

    # data inputs
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/datasets/expt-non-metals.csv",
        metavar="PATH",
        help="Path to main data set/training set",
    )
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument(
        "--val-path",
        type=str,
        metavar="PATH",
        help="Path to independent validation set",
    )
    valid_group.add_argument(
        "--val-size",
        default=0.0,
        type=float,
        metavar="FLOAT",
        help="Proportion of data used for validation",
    )
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--test-path",
        type=str,
        metavar="PATH",
        help="Path to independent test set"
    )
    test_group.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        metavar="FLOAT",
        help="Proportion of data set for testing",
    )

    # data embeddings
    parser.add_argument(
        "--fea-path",
        type=str,
        default="data/embeddings/matscholar-embedding.json",
        metavar="PATH",
        help="Element embedding feature path",
    )
    
    parser.add_argument(
        "--append_after",
        type=str,
        default="C",
        metavar="STR",
        help="ReacRoostC (C) or ReacRoostE (E)",
    )
    
    parser.add_argument(
        "--dim_red",
        action="store_true",
        help="Use or not use dimension reduction of the original embedding"
    )

    # dataloader inputs
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="INT",
        help="Number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        "--bsize",
        default=128,
        type=int,
        metavar="INT",
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--data-seed",
        default=0,
        type=int,
        metavar="INT",
        help="Seed used when splitting data sets (default: 0)",
    )
    parser.add_argument(
        "--sample",
        default=1,
        type=int,
        metavar="INT",
        help="Sub-sample the training set for learning curves",
    )

    # optimiser inputs
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="INT",
        help="Number of training epochs to run (default: 100)",
    )
    parser.add_argument(
        "--loss",
        default="L1",
        type=str,
        metavar="STR",
        help="Loss function if regression (default: 'L1')",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Specifies whether to use hetroskedastic loss variants",
    )
    parser.add_argument(
        "--optim",
        default="AdamW",
        type=str,
        metavar="STR",
        help="Optimizer used for training (default: 'AdamW')",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        default=3e-4,
        type=float,
        metavar="FLOAT",
        help="Initial learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-6,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer weight decay (default: 1e-6)",
    )

    # graph inputs
    
    parser.add_argument(
        "--model_class",
        default="ReacNN",
        type=str,
        metavar="STR",
        help="Number of hidden features for elements (default: 64)",
    )
    
    parser.add_argument(
        "--elem-fea-len",
        default=64,
        type=int,
        metavar="INT",
        help="Number of hidden features for elements (default: 64)",
    )
    parser.add_argument(
        "--n-graph",
        default=3,
        type=int,
        metavar="INT",
        help="Number of message passing layers (default: 3)",
    )

    # ensemble inputs
    parser.add_argument(
        "--ensemble",
        default=1,
        type=int,
        metavar="INT",
        help="Number models to ensemble",
    )
    name_group = parser.add_mutually_exclusive_group()
    name_group.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="STR",
        help="Name for sub-directory where models will be stored",
    )
    name_group.add_argument(
        "--data-id",
        default="roost",
        type=str,
        metavar="STR",
        help="Partial identifier for sub-directory where models will be stored",
    )
    parser.add_argument(
        "--run-id",
        default=0,
        type=int,
        metavar="INT",
        help="Index for model in an ensemble of models",
    )

    # restart inputs
    use_group = parser.add_mutually_exclusive_group()
    use_group.add_argument(
        "--fine-tune",
        type=str,
        metavar="PATH",
        help="Checkpoint path for fine tuning"
    )
    use_group.add_argument(
        "--transfer",
        type=str,
        metavar="PATH",
        help="Checkpoint path for transfer learning",
    )
    use_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint"
    )

    # task type
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--classification",
        action="store_true",
        help="Specifies a classification task"
    )
    task_group.add_argument(
        "--regression",
        action="store_true",
        help="Specifies a regression task"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model/ensemble",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model/ensemble"
    )

    # misc
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Disable CUDA"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log training metrics to tensorboard"
    )

    args = parser.parse_args(sys.argv[1:])

    if args.model_name is None:
        args.model_name = f"{args.data_id}_s-{args.data_seed}_t-{args.sample}"

    if args.regression:
        args.task = "regression"
    elif args.classification:
        args.task = "classification"
    else:
        args.task = "regression"

    args.device = (
        torch.device("cuda")
        if (not args.disable_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    return args

# =============================================================================
# ORIGINAL
# =============================================================================
# if __name__ == "__main__":
#     args = input_parser()

#     print(f"The model will run on the {args.device} device")
#     main(**vars(args))

if __name__ == "__main__":
    
#Running a task on a single train/test split
    args = input_parser() #get default parameters
    args = vars(args) #convert args into a dict
    args['train'] = True
    args['evaluate'] = True
    args['epochs'] = 2
    args['model_class'] = "ReacRoost"
    args['fea_path'] = 'embeddings/onehot-embedding.json'
    # args['fea_path'] = '/Users/korotkevich/Desktop/Experiments paper/code repo/embeddings/magpie_features.json'
    args['log'] = True
    args['elem_fea_len'] = 112
    args['append_after'] = "C"
    args['dim_red'] = False
    
    data_path = "data/D1/d1_f_rho.csv"
    args['data_path'] = data_path
    # print(args)
    main(**args)
    
#Running a cross-validation task
    # args = input_parser()
    # args = vars(args)
    # args['train'] = True
    # args['evaluate'] = True
    # args['epochs'] = 2
    # args['ensemble'] = 1
    # args['model_class'] = "ReacElemNet"
    # args['fea_path'] = '/Users/korotkevich/Desktop/Experiments paper/embeddings/onehot-embedding.json'
    # # args['fea_path'] = '/Users/korotkevich/Desktop/Experiments paper/code repo/embeddings/onehot-embedding.json'
    # args['log'] = True
    # args['elem_fea_len'] = 112
    # args['append_after'] = "C"
    # args['dim_red'] = False    

    # path_folds = '/Users/korotkevich/Desktop/Experiments paper/code repo/data/Dm/Folds/Ev/'
    # files = os.listdir(path_folds)
    # files.sort()
    
    # i = 0
    # while i < len(files):
        
    #     print(f'FOLD {i//2}')
        
    #     test_path = os.path.join(path_folds, files[i])
    #     train_path = os.path.join(path_folds, files[i+1])
    
    #     args['data_path'] = train_path
    #     args['test_path'] = test_path
    #     args['model_name'] = args['model_class'] + str(i//2)
    
            
    #     main(**args)

    #     i += 2
    
           
    
    # main(**args)
    
# http://localhost:6006/#timeseries
# args['transfer'] = '/Users/korotkevich/Desktop/Roost_tests/all/models/roost_10_epochs/checkpoint-r0.pth.tar'
# args['fine_tune'] = '/Users/korotkevich/Desktop/Roost_tests/all/models/roost_10_epochs/checkpoint-r0.pth.tar'
# args['batch_size'] = 256
# args['data_seed'] = 42
# args['test_size'] = 0.1
# print(args)
