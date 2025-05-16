import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_features, hidden_dims, n_classes, dropout=0.0,**kwargs):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_features, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.layers.append(nn.Linear(hidden_dims[-1], n_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x,_,**kwargs):
        x = self.dropout(x)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x