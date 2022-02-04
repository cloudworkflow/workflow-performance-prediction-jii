import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.lstm=nn.LSTM(config.n_feat,config.hidden,dropout=config.dropout,num_layers=config.num_layers)
        self.maxpool=nn.MaxPool1d(config.pooldim)
        self.fc=nn.Linear((config.hidden//config.pooldim)*config.num_task,config.num_classes)
    def forward(self,x):
        out=x.permute(1,0,2)
        out, _ = self.lstm(out)
        out=out.permute(1,0,2)
        #out = torch.tanh(out)
        out = self.maxpool(out)
        #out = torch.tanh(out)
        out=out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out