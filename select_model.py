import torch
class DAGTransformerConfig(object):
    def __init__(self):
        self.model_name='DAGTransformer'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.dropout = 0.3                                              
        self.num_classes = 3                                            
        self.num_epochs = 500                                          
        self.batch_size = 500                                         
        self.num_task = 7                                             
        self.learning_rate = 1e-4     
        self.n_feat = 34
        self.hidden_dim = 1024
        self.num_head = 8
        self.num_encoder = 6
        self.d_k=512
        self.res_num_layer=4
        self.structure=True

class CNNConfig(object):
    def __init__(self):
        self.model_name = 'CNN'
        self.n_feat = 34
        self.num_task = 7      
        self.outdim = 512
        self.num_epochs = 3000
        self.num_classes = 3
        self.pooldim = 3
        self.dropout = 0.3
        self.batch_size = 500
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.learning_rate = 1e-3

class LSTMConfig(object):
    def __init__(self):
        self.model_name = 'LSTM'
        self.n_feat=34
        self.num_task=7
        self.batch_size=500
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.learning_rate=1e-3
        self.num_epochs=500
        self.num_classes=3
        self.num_layers=6
        self.hidden=1024
        self.dropout=0.5
        self.pooldim = 3
        


class GCNConfig(object):
    def __init__(self):
        self.model_name = 'GCN'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_feat=34
        self.dropout = 0.5                                             
        self.num_classes = 3                                           
        self.num_epochs = 15000                                                                           
        self.learning_rate = 5e-3    

class VanillaTransformerConfig(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'VanillaTransformer'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.dropout = 0.5                                              
        self.num_classes = 3                                           
        self.num_epochs = 100                                          
        self.batch_size = 500                                          
        self.num_task = 7                                             
        self.learning_rate = 1e-4                                      
        self.n_feat = 34
        self.hidden = 1024
        self.num_head = 2
        self.num_encoder = 6

def select_model_exp1():
    
    config=DAGTransformerConfig()
    return config


def select_model_exp2(model_name):
    if model_name=='DAGTransformer':
        config=DAGTransformerConfig()
        config.num_epochs=100
        return config
        
    if model_name=='CNN':
        config=CNNConfig()
        config.num_epochs=100
        config.learning_rate=1e-4
        return config
    
    if model_name=='LSTM':
        config=LSTMConfig()
        config.learning_rate=1e-4
        config.num_epochs=100
        return config
    if model_name=='VanillaTransformer':
        config=VanillaTransformerConfig()
        return config


def select_model_exp3(model_name):
    if model_name=='DAGTransformer':
        config=DAGTransformerConfig()
        return config
    if model_name=='GCN':
        config=GCNConfig()
        return config
    if model_name=='CNN':
        config=CNNConfig()
        return config
    if model_name=='LSTM':
        config=LSTMConfig()
        return config
    if model_name=='VanillaTransformer':
        config=VanillaTransformerConfig()
        return config


    
        



    
