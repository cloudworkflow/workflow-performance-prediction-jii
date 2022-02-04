import torch
import torch.nn as nn
from torch.nn import  TransformerEncoderLayer
import copy

class Positional_Encoding(nn.Module):
    def __init__(self, n_feat, num_task, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / n_feat)) for i in range(n_feat)] for pos in range(num_task)])
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out

class VanillaTransformerModel(nn.Module):
    def __init__(self, config):
        super(VanillaTransformerModel, self).__init__()
        
        self.postion_embedding = Positional_Encoding(config.n_feat, config.num_task, config.dropout, config.device)
        self.encoder = TransformerEncoderLayer(config.n_feat, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])

        self.fc1 = nn.Linear(config.num_task * config.n_feat, config.num_classes)
        
    def forward(self, x):
        out = x
        #out=self.transform_shape(out)
        out = self.postion_embedding(out)
        out=out.permute(1,0,2)
        
        
        for encoder in self.encoders:
            out = encoder(out)
        out=out.permute(1,0,2)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out