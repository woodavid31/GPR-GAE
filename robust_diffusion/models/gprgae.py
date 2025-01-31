import torch
import copy
import torch.nn.functional as F
import numpy as np
from torch_sparse import SparseTensor
import torch.nn as nn

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from copy import deepcopy
from torch_geometric.nn.conv import MessagePassing



def coefficient_init(K, Init='Random', alpha = 0.75,Gamma = 1e-2):
        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS','Custom']
        if Init == 'SGC':
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha)**np.arange(K + 1)
            TEMP[-1] = (1 - alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma*np.ones(K + 1)
        elif Init == 'Custom':
            TEMP = 0.0 * np.ones(K + 1)

        return Parameter(torch.tensor(TEMP))


class GPRGAE(MessagePassing):
    def __init__(self, n_features, hidden, K,dropout_link,dropout_MLP,self_loop = False,activation_str = 'relu',concat_activation_str = 'relu',norm = False,elevate = True, **kwargs):
        super(GPRGAE, self).__init__(aggr='add')

        self.K = K
        self.coefficients = nn.ParameterList([coefficient_init(k) for k in range(K+1)])


        self.dropout_link = dropout_link
        self.dropout_mlp = dropout_MLP
        self.Sigmoid = nn.Sigmoid()
        self.activation_str = activation_str
        self.concat_activation_str = concat_activation_str
        self.self_loop = self_loop
        self.hidden = hidden

        ####Node embedding
        self.lin1 = nn.Linear(n_features, 1*hidden)
        self.lin2 = nn.Linear(1*hidden, 1*hidden)
        ####link pred
        self.lplin = nn.Linear(hidden*(K+1)*2,hidden*4)
        self.lplin2 = nn.Linear(hidden*4,1)
        self.lplin3 = nn.Linear(hidden*K,hidden*1)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def activation(self,x):
        if self.activation_str == 'relu':
            return F.relu(x)
        elif self.activation_str == 'elu':
            return F.elu(x)
        elif self.activation_str == 'leakyrelu':
            return F.leaky_relu(x)
        elif self.activation_str == 'sigmoid':
            return F.sigmoid(x)
        elif self.activation_str == 'tanh':
            return F.tanh(x)

    def concat_activation(self,x):
        if self.concat_activation_str == 'relu':
            return F.relu(x)
        elif self.concat_activation_str == 'elu':
            return F.elu(x)
        elif self.concat_activation_str == 'leakyrelu':
            return F.leaky_relu(x)
        elif self.concat_activation_str == 'sigmoid':
            return F.sigmoid(x)
        elif self.concat_activation_str == 'tanh':
            return F.tanh(x)
        

    def encode(self, data, adj):
        if isinstance(adj, SparseTensor):
            row, col, edge_weight = adj.t().coo()
            edge_index = torch.stack([row, col], dim=0)
        elif isinstance(adj, tuple):
            edge_index, edge_weight = adj
        
        if isinstance(edge_index, SparseTensor):
            row, col, edge_attr = edge_index.t().coo()
            edge_index = torch.stack([row, col], dim=0)

        # normalize adj
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes=data.size(0), add_self_loops=self.self_loop, dtype=data.dtype)


        data = F.dropout(data, p=self.dropout_mlp, training=self.training)
        data = self.lin1(data)

        powers = data.unsqueeze(0)
        gprs = [data]

        for k in range(1,self.K+1):
            data = self.propagate(edge_index, x=data, norm=edge_weight)
            powers = torch.cat((powers,data.unsqueeze(0)),dim = 0)
            cur_gpr = (self.coefficients[k].unsqueeze(-1).unsqueeze(-1)*powers).sum(dim = 0)
            gprs.append(cur_gpr)
        stack = torch.cat(gprs,dim = -1)
        
        return stack
    
    def decode(self, stack,edges):

        data = torch.cat((stack[edges[0]],stack[edges[1]]),dim = -1).float()
        #data = F.dropout(data,p = self.dropout_link,training = self.training)
        data = self.activation(data)
        data = self.lplin(data)
        data = F.dropout(data,p = self.dropout_link,training = self.training)
        data = self.activation(data)
        data = self.lplin2(data)

        lp = self.Sigmoid(data)
        return lp


    def forward(self, data, adj,batch = False):
        if isinstance(adj, SparseTensor):
            row, col, edge_weight = adj.t().coo()
            edge_index = torch.stack([row, col], dim=0)
        elif isinstance(adj, tuple):
            edge_index, edge_weight = adj
        
        if isinstance(edge_index, SparseTensor):
            row, col, edge_attr = edge_index.t().coo()
            edge_index = torch.stack([row, col], dim=0)

        for i in range(5): ## multi step purification
            prev = edge_weight.sum()
            edge_weight = self.link_prediction(data, (edge_index,edge_weight),edge_index,batch = batch).squeeze() + self.link_prediction(data, (edge_index,edge_weight),edge_index[[1,0]],batch = batch).squeeze()
            edge_weight /= 2

            if torch.abs(prev-edge_weight.sum())/prev <= 0.0001:
                break
        
        logits = self.gnn(data, (edge_index,edge_weight))

        return logits



    def link_prediction(self, data, adj,edges,batch = False):
        stack = self.encode(data,adj)
        
        if batch:
            lp = torch.ones_like(edges[0], dtype=torch.float)
            batch_size = 1024
            edge_num = edges.size(1)
            split_num = edge_num//batch_size
            for j in range(split_num+1):
                if j != split_num:
                    cur_edge_index = edges[:,j*batch_size:(j+1)*batch_size]
                    lp[j*batch_size:(j+1)*batch_size] = self.decode(stack,cur_edge_index).squeeze()
                else:
                    cur_edge_index = edges[:,j*batch_size:]
                    lp[j*batch_size:] = self.decode(stack,cur_edge_index).squeeze()
        else:
            lp = self.decode(stack,edges)

        return lp
    
    
    
