
from typing import Any, Dict, Union
import gc
import time
import torch.nn as nn
from copy import deepcopy
import argparse
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from robust_diffusion.models.gcn import GCN
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np

from robust_diffusion.data import prep_graph, split_inductive, filter_data_for_idx, count_edges_for_idx
from robust_diffusion.attacks import create_attack
from robust_diffusion.helper.io import Storage
from robust_diffusion.models import create_model,MLP
from robust_diffusion.models.gprgnn import GPR_prop
from robust_diffusion.train import train_inductive
from robust_diffusion.helper.utils import accuracy, calculate_loss
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def auc_ap(lp,lp_n):
    y_scores = torch.cat([lp.detach(), lp_n.detach()])
    y_true = torch.cat([torch.ones(lp.size(0)), torch.zeros(lp_n.size(0))])

    y_scores = y_scores.cpu().numpy()
    y_true = y_true.cpu().numpy()

    auc_score = roc_auc_score(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)

    return auc_score,ap_score


def mask_adjacency(edge_idx, mask_ratio):
    num_edges = int(edge_idx.size(1)/2)
    edge_idx_half = edge_idx[:,:num_edges]
    indices = np.random.permutation(num_edges)

    train_size = int(num_edges * (1-mask_ratio))

    train_edges = edge_idx_half[:, indices[:train_size]]
    val_edges = edge_idx_half[:, indices[train_size:]]

    train_edges = torch.cat((train_edges,train_edges[[1,0]]),dim = -1)
    val_edges = torch.cat((val_edges,val_edges[[1,0]]),dim = -1)

    return train_edges, val_edges

def split_edges(edge_idx, val_ratio=0.05, test_ratio=0.1):
    num_edges = edge_idx.size(1)
    indices = np.random.permutation(num_edges)

    # Determine split indices
    test_size = int(num_edges * test_ratio)
    val_size = int(num_edges * val_ratio)

    test_edges = edge_idx[:, indices[:test_size]]
    val_edges = edge_idx[:, indices[test_size:test_size + val_size]]
    train_edges = edge_idx[:, indices[test_size + val_size:]]

    train_edges = torch.cat((train_edges,train_edges[[1,0]]),dim = -1)
    val_edges = torch.cat((val_edges,val_edges[[1,0]]),dim = -1)
    test_edges = torch.cat((test_edges,test_edges[[1,0]]),dim = -1)

    return train_edges, val_edges, test_edges


def attack_params(attack_name):
    if attack_name == "PRBCD":
        train_attack_params = {'epochs': 20, 
                    'fine_tune_epochs': 0,
                    'keep_heuristic': 'WeightOnly',
                    'search_space_size': 1_000_000,
                    'loss_type': 'tanhMargin',
                    'with_early_stopping': False,
                    }
        val_attack_params = {'epochs': 20, 
                        'fine_tune_epochs': 0,
                        'keep_heuristic': 'WeightOnly',
                        'search_space_size': 1_000_000,
                        'loss_type': 'tanhMargin',
                        'with_early_stopping': False,
                        }
        test_attack_params = {'epochs':500,
            'fine_tune_epochs':100,
            'lr_factor':100,
            'keep_heuristic':"WeightOnly",
            'search_space_size':10_000,
            'do_synchronize':True,
            'binary_attribute':False,
            'loss_type':"tanhMargin"}
        
    elif attack_name == "LRBCD":
        train_attack_params = {'epochs': 20, 
                    'fine_tune_epochs': 0,
                    'keep_heuristic': 'WeightOnly',
                    'search_space_size': 1_000_000,
                    'loss_type': 'tanhMargin',
                    'with_early_stopping': False,
                    'lr_factor': 2000
                    }
        val_attack_params = {'epochs': 20, 
                        'fine_tune_epochs': 0,
                        'keep_heuristic': 'WeightOnly',
                        'search_space_size': 1_000_000,
                        'loss_type': 'tanhMargin',
                        'with_early_stopping': False,
                        'lr_factor': 2000
                        }
        test_attack_params = {'epochs':400,
            'fine_tune_epochs':0,
            'lr_factor':100,
            'keep_heuristic':"WeightOnly",
            'search_space_size':50_000,
            'do_synchronize':True,
            'with_early_stopping': False,
            'local_factor':0.5,
            'binary_attribute':False,
            'projection_type': "Greedy",
            'loss_type':"Margin"}
        
        
    return train_attack_params, val_attack_params, test_attack_params
def get_model_params(model_name):
    print(model_name)
    if model_name == 'GCN':
        model_params = dict(
            label="GCN",
            model="GCN",
            do_cache_adj_prep = False,
            n_filters = 64,
            dropout = 0.5,
            svd_params = None,
            jaccard_params = None,
            gdc_params = None
        )
    if model_name == "GAT":
        model_params = dict(
            label="GAT",
            model="GAT",
            hidden_dim = 64,
            dropout = 0.5
        )
    elif model_name == "GPRGNN":
        model_params = dict(
                label="GPRGNN",
                model="GPRGNN",
                propagation="GPR_prop",
                drop_GPR = "attr",
                K=10,
                hidden = 64,
                dropout_NN=0.2,
                dropout_GPR = 0 
            )
    elif model_name == "APPNP":
        model_params = dict(
                label="APPNP",
                model="GPRGNN",
                drop_GPR = "attr",
                K=10,
                hidden = 64,
                alpha = 0.1,
                propagation = "PPNP",
                dropout_NN=0.5,
                dropout_GPR = 0   
            )
    elif model_name == "GPRGAE":
        model_params = dict(
        label=model_name,
        model=model_name,
        hidden = 128,
        K = 7,
        dropout_link = 0,
        dropout_MLP = 0.7,
        self_loop = False,
        activation_str = 'elu',
        concat_activation_str = 'elu',
        elevate = True
        )

    return model_params
