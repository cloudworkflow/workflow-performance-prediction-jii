import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from .DAG_Transformer_Encoder_Layer import DAGTransformerEncoderLayer as encoder_layer


class DAGTransformerEncoder(nn.Module):
    def __init__(self, d_k, num_head, hidden_dim, dropout, num_layer):
        super(DAGTransformerEncoder, self).__init__()
        self.encoder_layer=encoder_layer(d_k, num_head, hidden_dim, dropout)
        self.encoder=nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(num_layer)])
    def forward(self, out, attn_mask=None):
        for _ in self.encoder:
            out = _(out, src_mask=attn_mask)
        return out



class resnet_layer(nn.Module):
    def __init__(self,n_feat,d_k,kernel_size,dropout):
        super(resnet_layer,self).__init__()
        self.conv1=nn.Conv1d(in_channels=n_feat,out_channels=d_k,kernel_size=kernel_size)
        self.conv2=nn.Conv1d(in_channels=d_k,out_channels=n_feat,kernel_size=kernel_size)
        self.dropout=nn.Dropout(dropout)


    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out+=x
        out=self.dropout(x)
        return out


class DAGTransformer(nn.Module):
    def __init__(self, config):
        super(DAGTransformer, self).__init__()
        self.structure=config.structure
        self.resnet1=resnet_layer(config.n_feat,config.d_k,1,config.dropout)
        if self.structure==True:
            self.resnet2=resnet_layer(config.n_feat,config.d_k,1,config.dropout)
        self.conv=nn.Conv1d(in_channels=config.n_feat,out_channels=config.d_k,kernel_size=1)
        self.res1=nn.ModuleList([
            copy.deepcopy(self.resnet1)
            for _ in range(config.res_num_layer)     
        ])
        if self.structure==True:
            self.res2=nn.ModuleList([
                copy.deepcopy(self.resnet2)
                for _ in range(config.res_num_layer)
            ])
        self.encoder = DAGTransformerEncoder(config.d_k, config.num_head, config.hidden_dim, config.dropout, config.num_encoder)
        self.avgpool=nn.AdaptiveAvgPool2d((config.d_k,1))
        self.fc1 = nn.Linear(config.d_k, 3)
    
    def forward(self,data,pos,mask):
        out = data
        out=out.permute(0,2,1)
        for resnet in self.res1:
            out = resnet(out)
        if self.structure==True:
            out1= pos
            attn_mask=mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask != 0, float(0.0))
            out1=out1.permute(0,2,1)
            for resnet in self.res2:
                out1=resnet(out1)
            out=self.conv(out+out1)
        else:
            out=self.conv(out)
        
        out=out.permute(2,0,1)
        
        out = self.encoder(out,attn_mask=attn_mask if self.structure==True else None)
        out=out.permute(1,2,0)
        out=F.relu(out)
        out=self.avgpool(out)
        out = out.squeeze(-1)
        out = self.fc1(out)
        return out
