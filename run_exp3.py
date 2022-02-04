import argparse
from pyexpat import model
from preprocess import preprocess_data_exp23_dag,preprocess_data_exp23, preprocess_data_exp23_GNN_bidir, preprocess_data_exp23_GNN_unidir
from select_model import select_model_exp3
from models.DAG_Transformer import DAGTransformer
from models.CNN import CNNModel
from models.LSTM import LSTMModel
from models.Vanilla_Transformer import VanillaTransformerModel
from models.GCN import GCNModel
from train_model_dag import train
from train_model_vanilla import train as train_vanilla
from train_model_gnn import train as train_gnn
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', required=True)#DAGTransformer, CNN, LSTM, VanillaTransformer, GCN
parser.add_argument('--split', default='split6_2_2')
parser.add_argument('--GCN_mode',default='bidirect')
opt = parser.parse_args()

if opt.model_name !='DAGTransformer' and opt.model_name !='CNN' and opt.model_name !='LSTM' and opt.model_name !='VanillaTransformer' and opt.model_name!='GCN':
    raise AssertionError('model should be DAGTransformer/CNN/LSTM/VanillaTransformer/GCN')
model_name=opt.model_name

if opt.split !='split9_05_05' and opt.split !='split8_1_1' and opt.split !='split6_2_2' :
    raise AssertionError('split should be split9_05_05/split8_1_1/split6_2_2')
split=opt.split
if opt.GCN_mode !='bidirect' and opt.GCN_mode!='unidirect':
    raise AssertionError('GCN_mode should be bidirect/unidirect')



if model_name!='GCN':
    if model_name=='DAGTransformer':
        config=select_model_exp3(model_name)
        train_data, val_data, test_data=preprocess_data_exp23_dag(split)
    else:
        config=select_model_exp3(model_name)
        train_data, val_data, test_data=preprocess_data_exp23(split)
    train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=config.batch_size,num_workers=2,
                                        shuffle=False)
    val_loader=torch.utils.data.DataLoader(dataset=val_data,batch_size=config.batch_size,num_workers=2,
                                        shuffle=False)
    test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=config.batch_size,num_workers=2,
                                        shuffle=False)
else:
    config=select_model_exp3('GCN')
    GCN_mode=opt.GCN_mode
    if GCN_mode=='bidirect':
        data=preprocess_data_exp23_GNN_bidir(split)
        
    else:
        data=preprocess_data_exp23_GNN_unidir(split)
    

if __name__=='__main__':
    if model_name=='DAGTransformer':
        model=DAGTransformer(config).to(config.device)
        train(config, model, train_loader, val_loader, test_loader)
    elif model_name=='LSTM':
        model=LSTMModel(config).to(config.device)
        train_vanilla(config, model, train_loader, val_loader, test_loader)
    elif model_name=='CNN':
        model=CNNModel(config).to(config.device)
        train_vanilla(config, model, train_loader, val_loader, test_loader)
    elif model_name=='VanillaTransformer':
        model=VanillaTransformerModel(config).to(config.device)
        train_vanilla(config, model, train_loader, val_loader, test_loader)
    elif model_name=='GCN':
        model=GCNModel(config).to(config.device)
        train_gnn(config,model,data)

    