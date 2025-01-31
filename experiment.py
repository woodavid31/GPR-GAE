from typing import Any, Dict, Union
import torch.nn as nn
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from robust_diffusion.models.gcn import GCN
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor
import numpy as np
from robust_diffusion.data import prep_graph, split_inductive, filter_data_for_idx, count_edges_for_idx
from robust_diffusion.attacks import create_attack
from robust_diffusion.models import create_model
from robust_diffusion.train import train_inductive
from robust_diffusion.helper.utils import accuracy
import warnings
from tools import *
import argparse
import os
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default='citeseer')
parser.add_argument('--attack', type=str,default='PRBCD')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--search_space', type=int, default=10000)
parser.add_argument("--no_print", action='store_true', default=False)
parser.add_argument("--adaptive", action='store_true', default=False)
parser.add_argument('--epsilon', type=float, default=1.5)
parser.add_argument('--mask', type=float, default=0.2)
parser.add_argument('--reweight', type=float, default=3)
parser.add_argument('--surrogate', type=str,default='GCN')
parser.add_argument('--K', type=int, default=7)


args = parser.parse_args()


# default params
n_per_class = 20
data_dir = './data'
make_undirected = True
binary_attr = False
balance_test = True

model_name = "GPRGAE"
model_params = get_model_params(model_name)
balance_test=True
attack = args.attack
_, _, test_attack_params = attack_params(attack)
test_attack_params['search_space_size'] = args.search_space
if args.adaptive:
    test_attack_params['loss_type'] = 'Margin' ## The better working loss type for adaptively attacking GPR-GAE

dataset = args.dataset
surrogate_model = args.surrogate
data_device = args.device
device = args.device
seed = args.seed
valid_epsilon = 0.3
val_ensemble = 10

train_epoch = args.epochs
robust_epsilon = args.epsilon
reweight = args.reweight

mask_ratio = args.mask 




model_params['K'] = args.K
model_params['hidden'] = 128




print("########################################################################################################################################################################################################################################")
print("########################################################################################################################################################################################################################################")
print(vars(args))

torch.manual_seed(seed)
np.random.seed(seed)

torch.set_default_device(device)

self_model_params = get_model_params(surrogate_model)

########################### LOAD DATA ###########################
graph = prep_graph(dataset, data_device, dataset_root=data_dir, make_undirected=make_undirected,
                    binary_attr=binary_attr, return_original_split=dataset.startswith('ogbn'),
                    seed=seed)

attr_orig, adj_orig, labels = graph[:3]

########################## Create Model ###############################
hyperparams = dict(model_params)
self_hyperparams = dict(self_model_params)
n_features = attr_orig.shape[1]
n_classes = int(labels.max() + 1)
hyperparams.update({
    'n_features': n_features,
    'n_classes': n_classes
})
self_hyperparams.update({
    'n_features': n_features,
    'n_classes': n_classes
})
model = create_model(self_hyperparams).to(device)

########################### Split ########################################
if dataset == 'ogbn-arxiv':
    adj_orig = adj_orig.type_as(torch.ones(1, dtype=torch.float32, device=device)) # cast to float
    idx_train = graph[3]['train']
    idx_val = graph[3]['valid']
    idx_test = graph[3]['test']
    idx_unlabeled = np.array([], dtype=bool)
elif dataset == "chameleon":
    idx_train = graph[3]["trn_idx"]
    idx_val = graph[3]["val_idx"]
    idx_test = graph[3]["test_idx"]
    idx_unlabeled = []
else:
    idx_train, idx_unlabeled, idx_val, idx_test = split_inductive(labels.cpu().numpy(), n_per_class=n_per_class, seed = seed, balance_test=balance_test)

########################### handle data #####################
# delete 
attr_train, adj_train, labels_train, mapping_proj_train = filter_data_for_idx(attr_orig.clone(), adj_orig.clone(), labels,  np.concatenate([idx_train, idx_unlabeled]))
idx_train_train = mapping_proj_train[idx_train] # idx of training nodes in training graph
idx_unlabeled_train = mapping_proj_train[idx_unlabeled] # idx of unlabeled nodes in training graph
attr_val, adj_val, labels_val, mapping_proj_val = filter_data_for_idx(attr_orig.clone(), adj_orig.clone(), labels, np.concatenate([idx_train, idx_val, idx_unlabeled]))
idx_val_val = mapping_proj_val[idx_val] # idx of val nodes in val graph

########################## Train Surrogate Model ###############################
surrogate_train_params = dict(
    lr=1e-2,
    weight_decay=1e-3,
    patience=200,
    max_epochs=3000
)

train_inductive(model=model.to(device), 
                attr_training=attr_train.to(device), 
                attr_validation=attr_val.to(device), 
                adj_training= adj_train.to(device), 
                adj_validation=adj_val.to(device),
                labels_training=labels_train.to(device),
                labels_validation=labels_val.to(device),
                idx_train=idx_train_train,
                idx_val=idx_val_val,
                **surrogate_train_params)
model.eval()

logits_clean_test = model.to(device)(attr_orig.to(device), adj_orig.to(device))
test_accuracy_clean = accuracy(logits_clean_test.cpu(), labels.cpu(), idx_test)      
print("test accuracy clean:",test_accuracy_clean)


########################### SELF-TRAINING ####################
gprgae = create_model(hyperparams).to(device)
############################### REST ###################################
optimizer = torch.optim.Adam(gprgae.parameters(), lr=1e-2,weight_decay = 1e-4)
############################## n_pertubations ##########################

idx_train_train = np.concatenate([idx_train_train, idx_unlabeled_train])
n_train_edges = count_edges_for_idx(adj_train, idx_train_train) # num edges connected to train nodes
m_train = int(n_train_edges) / 2
n_perturbations_train = int(round(1 * m_train))

n_val_edges = count_edges_for_idx(adj_val, idx_val_val) # num edges connected to val nodes
m_val = int(n_val_edges) / 2
n_perturbations_val = int(round(valid_epsilon * m_val))

n_test_edges = count_edges_for_idx(adj_orig, idx_test) # num edges connected to val nodes
m_test = int(n_test_edges) / 2

############################### Train Loop ###################################

if args.adaptive:
    attack_target = gprgae
else:
    attack_target = model


best_loss=np.inf

##### rearrange the train edges into a concatenation of upper triangle and lower triangle
_, edge_idx_h, _ = GCN.parse_forward_input(attr_train, adj_train, None, None, None, None)
edge_idx_h = edge_idx_h[:,edge_idx_h[0]<edge_idx_h[1]]
edge_idx = torch.cat((edge_idx_h,edge_idx_h[[1,0]]),dim = -1)


##############Create Validation Attacked Graph################################              
torch.cuda.empty_cache()

_, edge_idx_val, _ = GCN.parse_forward_input(attr_val, adj_val, None, None, None, None)

neg_val_list = []
for i in range(val_ensemble):
    neg_val_list.append(negative_sampling(edge_idx_val,adj_val.size(dim = 0),n_perturbations_val*(i+1),force_undirected = True))


#######################################################################################

for it in tqdm(range(train_epoch), desc="Training Progress", leave=True):

    torch.cuda.empty_cache()


    unmasked_train_edges,masked_train_edges = mask_adjacency(edge_idx, mask_ratio = mask_ratio) 
    neg_edge_idx = negative_sampling(edge_idx,adj_train.size(dim = 0),int(n_perturbations_train * robust_epsilon),force_undirected = True)
    unmasked_neg_edges,masked_neg_edges = mask_adjacency(neg_edge_idx, mask_ratio = mask_ratio)


    gprgae.train()

    optimizer.zero_grad()

    cur_train_edges = torch.cat((unmasked_train_edges,unmasked_neg_edges),dim = -1)
    cur_weight = torch.ones_like(cur_train_edges[0], dtype=torch.float)
    train_weight = torch.empty((len(unmasked_train_edges[0])//2)).uniform_(1,1*reweight)
    neg_weight = torch.empty((len(unmasked_neg_edges[0])//2)).uniform_(1,1*reweight)
    cur_weight = torch.cat([train_weight,train_weight,neg_weight,neg_weight],dim = -1)

    # Combine positive and negative edges into one tensor
    combined_edges = torch.cat((edge_idx, neg_edge_idx), dim=-1)

    # Perform link prediction for all edges (positive + negative) at once
    lp_combined = gprgae.link_prediction(attr_train, (cur_train_edges, cur_weight), combined_edges)

    # Split predictions back into positive and negative
    num_pos = edge_idx.size(1)
    lp, lp_n = lp_combined[:num_pos], lp_combined[num_pos:]

    # Compute binary cross-entropy loss for positive and negative edges
    pos_loss = F.binary_cross_entropy(lp, torch.ones_like(lp))
    neg_loss = F.binary_cross_entropy(lp_n, torch.zeros_like(lp_n))

    # Compute regularization loss for combined edges
    inverse_combined = combined_edges[[1, 0]]  # Swap rows for all edges
    original = lp_combined
    inverse = gprgae.link_prediction(attr_train, (cur_train_edges, cur_weight), inverse_combined)
    reg_loss = F.mse_loss(original, inverse)

    # Combine losses
    loss = pos_loss + neg_loss + 0.2 * reg_loss

    
    #print("[pos loss: ",pos_loss.item()," neg loss: ",neg_loss.item(),"] [lp mean: ",lp.mean().item(),"lp_n mean: ",lp_n.mean().item(),"]")
    
    loss.backward()
    optimizer.step()


    # val step 
    if it % 3 == 0:
        with torch.no_grad():
            # validation
            gprgae.eval()
            val_auc,val_ap = 0,0
            for i in range(val_ensemble):
                neg_edge_idx = neg_val_list[i]
                # Combine validation edges (positive and negative)
                cur_val_edges = torch.cat((edge_idx_val, neg_edge_idx), dim=-1)

                # Perform link prediction for all validation edges (positive + negative)
                lp_combined = gprgae.link_prediction(attr_val,(cur_val_edges, torch.ones_like(cur_val_edges[0], dtype=torch.float)),cur_val_edges)

                # Split predictions into positive and negative parts
                num_pos = edge_idx_val.size(1)
                lp, lp_n = lp_combined[:num_pos], lp_combined[num_pos:]

                # Compute binary cross-entropy loss for positive and negative predictions
                pos_loss = F.binary_cross_entropy(lp, torch.ones_like(lp))
                neg_loss = F.binary_cross_entropy(lp_n, torch.zeros_like(lp_n))

                auc_score,ap_score = auc_ap(lp,lp_n)
                val_auc += auc_score
                val_ap += ap_score
            val_auc /= val_ensemble
            val_ap /= val_ensemble

            val_auc_ap = val_auc + val_ap

            if not args.no_print:
                print("[VAL] AUC:",val_auc,"AP:",val_ap)

            # save new best model and break if patience is reached
            if -(val_auc_ap) < best_loss:
                if not args.no_print:
                    print("Best model Updated!!!!!!!!!!!!-------------------------------------")
                best_loss = -(val_auc_ap)
                best_state = {key: value.cpu() for key, value in gprgae.state_dict().items()}


# restore the best validation state
gprgae.load_state_dict(best_state)
gprgae.eval()

torch.cuda.empty_cache()
###################################################################################################
# Adversarial Robustness Test Code
model.eval()
gprgae.gnn = model

logits_clean_test = model.to(device)(attr_orig.to(device), adj_orig.to(device))
test_accuracy_clean = accuracy(logits_clean_test.cpu(), labels.cpu(), idx_test)
print("Clean test accuracy:",test_accuracy_clean)


adversary_test = create_attack(attack, attr=attr_orig, adj=adj_orig, labels=labels, model=attack_target, idx_attack=idx_test,
                device=device, data_device=data_device, binary_attr=binary_attr,
                make_undirected=make_undirected, **test_attack_params)

for ep in [0,0.1,0.25,0.5]:
    print("###################################################################################################")
    torch.cuda.empty_cache()
    adversary_test.attack(int(round(m_test*ep)))
    adj_adversary = adversary_test.adj_adversary

    logits_adv_test = model(attr_orig.to(device), adj_adversary.to(device))
    test_accuracy_adv = accuracy(logits_adv_test.cpu(), labels.cpu(), idx_test)

    print("Surrogate test accuracy:",test_accuracy_adv)

    gprgae.eval()
    
    with torch.no_grad():
        logits_adv_test = gprgae(attr_orig.to(device), adj_adversary.to(device),batch = True)
        test_accuracy_adv = accuracy(logits_adv_test.cpu(), labels.cpu(), idx_test)

    torch.cuda.empty_cache()


    print("Attacked gprgae test accuracy:",test_accuracy_adv)
    print("--------------------------------------------------------------------")
    del adj_adversary
    torch.cuda.empty_cache()



