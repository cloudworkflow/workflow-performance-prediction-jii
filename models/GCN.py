import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNModel(nn.Module):
    def __init__(self,config):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(34, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 3)
        self.dropout1=nn.Dropout(config.dropout)
        self.dropout2=nn.Dropout(config.dropout)
        self.dropout3=nn.Dropout(config.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.conv4(x, edge_index)
        return x