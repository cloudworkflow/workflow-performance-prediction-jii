import torch 
import torch.nn as nn
import torch.nn.functional as F
class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()
        self.conv=nn.Conv1d(in_channels=config.n_feat,out_channels=config.outdim,kernel_size=1)
        self.maxpool=nn.MaxPool1d(config.pooldim)
        self.fc=nn.Linear((config.outdim//config.pooldim)*config.num_task,config.num_classes)
        self.dropout=nn.Dropout(config.dropout)
    def forward(self,x):
        out=x.permute(0,2,1)
        out=F.relu(self.conv(out))
        out=self.dropout(out)
        out=out.permute(0,2,1)
        out=self.maxpool(out)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        return out