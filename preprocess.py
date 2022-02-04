
import pandas as pd
import numpy as np
from tqdm import tqdm

def find_pos(out_degree_matrix,num_nodes):
    stage=np.zeros(num_nodes)
    signal=True
    while signal:
        temp=stage.copy()
        for m in range(num_nodes):
            for n in range(num_nodes):
                if(out_degree_matrix[m,n]!=0):
                    stage[n]=max(stage[n],stage[m]+1)
        if (temp==stage).all():
            signal=False
    return stage

def create_position(pos,num_feat):
    
    pe = np.array([[[posit / (10000.0 ** (i // 2 * 2.0 / num_feat))  for i in range(num_feat)]for posit in posi] for posi in pos])
    pe[:,:,0::2]=np.sin(pe[:,:,0::2])
    pe[:,:,1::2]=np.cos(pe[:,:,1::2])
    return pe

def create_attn_mask(tensor,num_heads,num_nodes):
    mask=np.zeros(((tensor.shape[0]*num_heads),tensor.shape[1],tensor.shape[2]))
    for x in range(0,mask.shape[0],num_heads):
        mask[x:x+num_heads]=tensor[x//num_heads]+np.eye(num_nodes)
    return mask.reshape(tensor.shape[0],num_heads,tensor.shape[1],tensor.shape[2])

def prepare_data_exp1_dag(pred_task,pred_tgt,pred_mode):
    direct='./datasets_exp1/%s%s/%s'%(pred_task,pred_tgt,pred_mode)
    df_train=pd.read_csv(direct+'/train.csv')
    df_val=pd.read_csv(direct+'/val.csv')
    df_test=pd.read_csv(direct+'/test.csv')
    dag_direct='./datasets_exp1/%s%s/'%(pred_task,pred_tgt)
    df_dag_train=pd.read_csv(dag_direct+'/train_daginfo.csv')
    df_dag_val=pd.read_csv(dag_direct+'/val_daginfo.csv')
    df_dag_test=pd.read_csv(dag_direct+'/test_daginfo.csv')
    return df_train, df_val, df_test, df_dag_train, df_dag_val, df_dag_test

def prepare_data_exp1(pred_task,pred_tgt,pred_mode):
    direct='./datasets_exp1/%s%s/%s'%(pred_task,pred_tgt,pred_mode)
    df_train=pd.read_csv(direct+'/train.csv')
    df_val=pd.read_csv(direct+'/val.csv')
    df_test=pd.read_csv(direct+'/test.csv')
    
    return df_train, df_val, df_test

def prepare_data_exp23_dag(split):#split='split6_2_2','split8_1_1'，'split9_05_05'
    direct='./datasets_exp2_3/%s'%split
    df_train=pd.read_csv(direct+'/train.csv')
    df_val=pd.read_csv(direct+'/val.csv')
    df_test=pd.read_csv(direct+'/test.csv')
    df_dag_train=pd.read_csv(direct+'/train_daginfo.csv')
    df_dag_val=pd.read_csv(direct+'/val_daginfo.csv')
    df_dag_test=pd.read_csv(direct+'/test_daginfo.csv')
    return df_train, df_val, df_test, df_dag_train, df_dag_val, df_dag_test


def prepare_data_exp23(split):#split='split6_2_2','split8_1_1'，'split9_05_05'
    direct='./datasets_exp2_3/%s'%split
    df_train=pd.read_csv(direct+'/train.csv')
    df_val=pd.read_csv(direct+'/val.csv')
    df_test=pd.read_csv(direct+'/test.csv')
    return df_train, df_val, df_test
        

def preprocess_data_exp1_dag(pred_task,pred_tgt,pred_mode,num_feat=34,num_task=7,num_head=8):
    df_train, df_val, df_test, df_dag_train, df_dag_val, df_dag_test=prepare_data_exp1_dag(pred_task,pred_tgt,pred_mode)
    ##nodes features
    arr1=np.array(df_train.iloc[:,:-1])
    arr2=np.array(df_val.iloc[:,:-1])
    arr3=np.array(df_test.iloc[:,:-1])
    data=np.vstack((arr1.reshape(arr1.shape[0]*num_task,num_feat),arr2.reshape(arr2.shape[0]*num_task,num_feat),arr3.reshape(arr3.shape[0]*num_task,num_feat)))
    data=(data-data.min(0))/(data.max(0)-data.min(0)+1e-9)
    data1=data[:arr1.shape[0]*num_task,:]
    data2=data[arr1.shape[0]*num_task:arr1.shape[0]*num_task+arr2.shape[0]*num_task,:]
    data3=data[arr1.shape[0]*num_task+arr2.shape[0]*num_task:,:]  
    data1=data1.reshape(-1,num_task,num_feat)
    data2=data2.reshape(-1,num_task,num_feat)
    data3=data3.reshape(-1,num_task,num_feat)
    #####dag info
    dag1=df_dag_train.to_numpy().reshape(-1,num_task,num_task*2+1)
    dag2=df_dag_val.to_numpy().reshape(-1,num_task,num_task*2+1)
    dag3=df_dag_test.to_numpy().reshape(-1,num_task,num_task*2+1)
    dag1=dag1[:,:,1:]
    dag2=dag2[:,:,1:]
    dag3=dag3[:,:,1:]
    dagout1=dag1[:,:,:7]
    dagout2=dag2[:,:,:7]
    dagout3=dag3[:,:,:7]
    dagin1=dag1[:,:,7:]
    dagin2=dag2[:,:,7:]
    dagin3=dag3[:,:,7:]
    mask1=dagin1+dagout1
    mask2=dagin2+dagout2
    mask3=dagin3+dagout3
    pos1=np.zeros((arr1.shape[0],7))
    for x in range(pos1.shape[0]):
        pos1[x]=find_pos(dagout1[x],7)
    pos2=np.zeros((arr2.shape[0],7))
    for x in range(pos2.shape[0]):
        pos2[x]=find_pos(dagout2[x],7)
    pos3=np.zeros((arr3.shape[0],7))
    for x in range(pos3.shape[0]):
        pos3[x]=find_pos(dagout3[x],7)
    position1=create_position(pos1,num_feat)
    position2=create_position(pos2,num_feat)
    position3=create_position(pos3,num_feat)
    mask_train=create_attn_mask(mask1,num_heads=num_head,num_nodes=num_task)
    mask_val=create_attn_mask(mask2,num_heads=num_head,num_nodes=num_task)
    mask_test=create_attn_mask(mask3,num_heads=num_head,num_nodes=num_task)
    data1=np.array(data1,dtype=np.float32)
    data2=np.array(data2,dtype=np.float32)
    data3=np.array(data3,dtype=np.float32)
    position1=np.array(position1,dtype=np.float32)
    position2=np.array(position2,dtype=np.float32)
    position3=np.array(position3,dtype=np.float32)
    ######data
    train_data=[]
    for x in range(data1.shape[0]):
        train_data.append((data1[x],df_train.iloc[x,-1],position1[x],mask_train[x]))
    val_data=[]
    for x in range(data2.shape[0]):
        val_data.append((data2[x],df_val.iloc[x,-1],position2[x],mask_val[x]))
    test_data=[]
    for x in range(data3.shape[0]):
        test_data.append((data3[x],df_test.iloc[x,-1],position3[x],mask_test[x]))
    return train_data, val_data, test_data



def preprocess_data_exp1(pred_task,pred_tgt,pred_mode,num_feat=34,num_task=7,):
    df_train, df_val, df_test = prepare_data_exp1(pred_task,pred_tgt,pred_mode)
    ##nodes features
    arr1=np.array(df_train.iloc[:,:-1])
    arr2=np.array(df_val.iloc[:,:-1])
    arr3=np.array(df_test.iloc[:,:-1])
    data=np.vstack((arr1.reshape(arr1.shape[0]*num_task,num_feat),arr2.reshape(arr2.shape[0]*num_task,num_feat),arr3.reshape(arr3.shape[0]*num_task,num_feat)))
    data=(data-data.min(0))/(data.max(0)-data.min(0)+1e-9)
    data1=data[:arr1.shape[0]*num_task,:]
    data2=data[arr1.shape[0]*num_task:arr1.shape[0]*num_task+arr2.shape[0]*num_task,:]
    data3=data[arr1.shape[0]*num_task+arr2.shape[0]*num_task:,:]  
    data1=data1.reshape(-1,num_task,num_feat)
    data2=data2.reshape(-1,num_task,num_feat)
    data3=data3.reshape(-1,num_task,num_feat)
    data1=np.array(data1,dtype=np.float32)
    data2=np.array(data2,dtype=np.float32)
    data3=np.array(data3,dtype=np.float32)

    ######data
    train_data=[]
    for x in range(data1.shape[0]):
        train_data.append((data1[x],df_train.iloc[x,-1]))
    val_data=[]
    for x in range(data2.shape[0]):
        val_data.append((data2[x],df_val.iloc[x,-1]))
    test_data=[]
    for x in range(data3.shape[0]):
        test_data.append((data3[x],df_test.iloc[x,-1]))
    return train_data, val_data, test_data


def preprocess_data_exp23_dag(split,num_feat=34,num_task=7,num_head=8):
    df_train, df_val, df_test, df_dag_train, df_dag_val, df_dag_test=prepare_data_exp23_dag(split)
    ##nodes features
    arr1=np.array(df_train.iloc[:,:-1])
    arr2=np.array(df_val.iloc[:,:-1])
    arr3=np.array(df_test.iloc[:,:-1])
    data=np.vstack((arr1.reshape(arr1.shape[0]*num_task,num_feat),arr2.reshape(arr2.shape[0]*num_task,num_feat),arr3.reshape(arr3.shape[0]*num_task,num_feat)))
    data=(data-data.min(0))/(data.max(0)-data.min(0)+1e-9)
    data1=data[:arr1.shape[0]*num_task,:]
    data2=data[arr1.shape[0]*num_task:arr1.shape[0]*num_task+arr2.shape[0]*num_task,:]
    data3=data[arr1.shape[0]*num_task+arr2.shape[0]*num_task:,:]  
    data1=data1.reshape(-1,num_task,num_feat)
    data2=data2.reshape(-1,num_task,num_feat)
    data3=data3.reshape(-1,num_task,num_feat)
    #####dag info
    dag1=df_dag_train.to_numpy().reshape(-1,num_task,num_task*2+1)
    dag2=df_dag_val.to_numpy().reshape(-1,num_task,num_task*2+1)
    dag3=df_dag_test.to_numpy().reshape(-1,num_task,num_task*2+1)
    dag1=dag1[:,:,1:]
    dag2=dag2[:,:,1:]
    dag3=dag3[:,:,1:]
    dagout1=dag1[:,:,:7]
    dagout2=dag2[:,:,:7]
    dagout3=dag3[:,:,:7]
    dagin1=dag1[:,:,7:]
    dagin2=dag2[:,:,7:]
    dagin3=dag3[:,:,7:]
    mask1=dagin1+dagout1
    mask2=dagin2+dagout2
    mask3=dagin3+dagout3
    pos1=np.zeros((arr1.shape[0],7))
    for x in range(pos1.shape[0]):
        pos1[x]=find_pos(dagout1[x],7)
    pos2=np.zeros((arr2.shape[0],7))
    for x in range(pos2.shape[0]):
        pos2[x]=find_pos(dagout2[x],7)
    pos3=np.zeros((arr3.shape[0],7))
    for x in range(pos3.shape[0]):
        pos3[x]=find_pos(dagout3[x],7)
    position1=create_position(pos1,num_feat)
    position2=create_position(pos2,num_feat)
    position3=create_position(pos3,num_feat)
    mask_train=create_attn_mask(mask1,num_heads=num_head,num_nodes=num_task)
    mask_val=create_attn_mask(mask2,num_heads=num_head,num_nodes=num_task)
    mask_test=create_attn_mask(mask3,num_heads=num_head,num_nodes=num_task)
    data1=np.array(data1,dtype=np.float32)
    data2=np.array(data2,dtype=np.float32)
    data3=np.array(data3,dtype=np.float32)
    position1=np.array(position1,dtype=np.float32)
    position2=np.array(position2,dtype=np.float32)
    position3=np.array(position3,dtype=np.float32)
    ######data
    train_data=[]
    for x in range(data1.shape[0]):
        train_data.append((data1[x],df_train.iloc[x,-1],position1[x],mask_train[x]))
    val_data=[]
    for x in range(data2.shape[0]):
        val_data.append((data2[x],df_val.iloc[x,-1],position2[x],mask_val[x]))
    test_data=[]
    for x in range(data3.shape[0]):
        test_data.append((data3[x],df_test.iloc[x,-1],position3[x],mask_test[x]))
    return train_data, val_data, test_data


def preprocess_data_exp23(split,num_feat=34,num_task=7,):
    df_train, df_val, df_test = prepare_data_exp23(split)
    ##nodes features
    arr1=np.array(df_train.iloc[:,:-1])
    arr2=np.array(df_val.iloc[:,:-1])
    arr3=np.array(df_test.iloc[:,:-1])
    data=np.vstack((arr1.reshape(arr1.shape[0]*num_task,num_feat),arr2.reshape(arr2.shape[0]*num_task,num_feat),arr3.reshape(arr3.shape[0]*num_task,num_feat)))
    data=(data-data.min(0))/(data.max(0)-data.min(0)+1e-9)
    data1=data[:arr1.shape[0]*num_task,:]
    data2=data[arr1.shape[0]*num_task:arr1.shape[0]*num_task+arr2.shape[0]*num_task,:]
    data3=data[arr1.shape[0]*num_task+arr2.shape[0]*num_task:,:]  
    data1=data1.reshape(-1,num_task,num_feat)
    data2=data2.reshape(-1,num_task,num_feat)
    data3=data3.reshape(-1,num_task,num_feat)
    data1=np.array(data1,dtype=np.float32)
    data2=np.array(data2,dtype=np.float32)
    data3=np.array(data3,dtype=np.float32)


    ######data
    train_data=[]
    for x in range(data1.shape[0]):
        train_data.append((data1[x],df_train.iloc[x,-1]))
    val_data=[]
    for x in range(data2.shape[0]):
        val_data.append((data2[x],df_val.iloc[x,-1]))
    test_data=[]
    for x in range(data3.shape[0]):
        test_data.append((data3[x],df_test.iloc[x,-1]))
    return train_data, val_data, test_data



import torch
from torch_geometric.data import Data
def preprocess_data_exp23_GNN_unidir(split,num_feat=34,num_task=7):
    df_train, df_val, df_test, df_dag_train, df_dag_val, df_dag_test=prepare_data_exp23_dag(split)
    df=pd.concat((df_train,df_val,df_test),axis=0)
    feat=df.iloc[:,:-1].to_numpy().reshape(-1,num_feat)
    label=df.iloc[:,-1].to_numpy()
    labels=torch.zeros(df.shape[0]*num_task)
    for x in range(labels.shape[0]):
        if((x+1)%num_task==0):
            labels[x]=label[x//num_task]
    df_dag=pd.concat((df_dag_train,df_dag_val,df_dag_test),axis=0)
    edge_info=df_dag.to_numpy().reshape(-1,2*num_task+1)
    edge_info=edge_info[:,1:]
    out_mat=edge_info[:,:num_task]
    edge=torch.tensor([[],[]],dtype=torch.long)
    print('preparing data...')
    for src in tqdm(range(out_mat.shape[0])):
        for tgt in range(num_task):
            if(out_mat[src,tgt]!=0):
                edge=torch.cat((edge,torch.tensor([[src],[src+(tgt-src%num_task)]],dtype=torch.long)),1)
    self_loop=torch.tensor([[x for x in range(df.shape[0]*num_task)],[x for x in range(df.shape[0]*num_task)]],dtype=torch.long)
    edge=torch.cat((edge,self_loop),1)
    feat=(feat-feat.min(0))/(feat.max(0)-feat.min(0)+1e-9)
    feat=torch.tensor(feat,dtype=torch.float)
    Gdata = Data(x=feat, edge_index=edge)
    Gdata.labels=labels.long()
    Gdata.train_mask=torch.ByteTensor([False for x in range(df.shape[0]*num_task)]).bool()
    Gdata.val_mask=torch.ByteTensor([False for x in range(df.shape[0]*num_task)]).bool()
    Gdata.test_mask=torch.ByteTensor([False for x in range(df.shape[0]*num_task)]).bool()
    for x in range(df.shape[0]*num_task):
        if(x<df_train.shape[0]*num_task and (x+1)%num_task==0):
            Gdata.train_mask[x]=True
    
        elif(x<(df_train.shape[0]*num_task+df_val.shape[0]*num_task) and (x+1)%num_task==0):
            Gdata.val_mask[x]=True
        elif(x>=(df_train.shape[0]*num_task+df_val.shape[0]*num_task) and (x+1)%num_task==0):
            Gdata.test_mask[x]=True
    return Gdata


def preprocess_data_exp23_GNN_bidir(split,num_feat=34,num_task=7):
    df_train, df_val, df_test, df_dag_train, df_dag_val, df_dag_test=prepare_data_exp23_dag(split)
    df=pd.concat((df_train,df_val,df_test),axis=0)
    feat=df.iloc[:,:-1].to_numpy().reshape(-1,num_feat)
    label=df.iloc[:,-1].to_numpy()
    labels=torch.zeros(df.shape[0]*num_task)
    for x in range(labels.shape[0]):
        if((x+1)%num_task==0):
            labels[x]=label[x//num_task]
    df_dag=pd.concat((df_dag_train,df_dag_val,df_dag_test),axis=0)
    edge_info=df_dag.to_numpy().reshape(-1,2*num_task+1)
    edge_info=edge_info[:,1:]
    out_mat=edge_info[:,:num_task]
    edge=torch.tensor([[],[]],dtype=torch.long)
    print('preparing data...')
    for src in tqdm(range(out_mat.shape[0])):
        for tgt in range(num_task):
            if(out_mat[src,tgt]!=0):
                edge=torch.cat((edge,torch.tensor([[src],[src+(tgt-src%num_task)]],dtype=torch.long),
                           torch.tensor([[src+(tgt-src%num_task)],[src]],dtype=torch.long)),1)
    self_loop=torch.tensor([[x for x in range(df.shape[0]*num_task)],[x for x in range(df.shape[0]*num_task)]],dtype=torch.long)
    edge=torch.cat((edge,self_loop),1)
    feat=(feat-feat.min(0))/(feat.max(0)-feat.min(0)+1e-9)
    feat=torch.tensor(feat,dtype=torch.float)
    Gdata = Data(x=feat, edge_index=edge)
    Gdata.labels=labels.long()
    Gdata.train_mask=torch.ByteTensor([False for x in range(df.shape[0]*num_task)]).bool()
    Gdata.val_mask=torch.ByteTensor([False for x in range(df.shape[0]*num_task)]).bool()
    Gdata.test_mask=torch.ByteTensor([False for x in range(df.shape[0]*num_task)]).bool()
    for x in range(df.shape[0]*num_task):
        if(x<df_train.shape[0]*num_task and (x+1)%num_task==0):
            Gdata.train_mask[x]=True
        elif(x<(df_train.shape[0]*num_task+df_val.shape[0]*num_task) and (x+1)%num_task==0):
            Gdata.val_mask[x]=True
        elif(x>=(df_train.shape[0]*num_task+df_val.shape[0]*num_task) and (x+1)%num_task==0):
            Gdata.test_mask[x]=True
    return Gdata