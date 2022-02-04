import argparse
from preprocess import preprocess_data_exp1_dag,preprocess_data_exp1
from select_model import select_model_exp1
from models.DAG_Transformer import DAGTransformer
from train_model_dag import train
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_task', default=7)
parser.add_argument('--pred_tgt', default='CPU')
parser.add_argument('--pred_mode',default='PRIOR_ALL')
parser.add_argument('--use_DAG',default='T')
opt = parser.parse_args()

if opt.pred_task !='3' and opt.pred_task!='5' and opt.pred_task!='7':
    raise AssertionError('pred_task should be 3/5/7')
pred_task=opt.pred_task

if opt.pred_tgt !='CPU' and opt.pred_tgt!='MEM':
    raise AssertionError('pred_tgt should be CPU/MEM')
pred_tgt=opt.pred_tgt

if opt.pred_mode !='PRIOR_1' and opt.pred_mode!='PRIOR_ALL':
    raise AssertionError('pred_mode should be PRIOR_1/PRIOR_ALL')
pred_mode=opt.pred_mode

if opt.use_DAG =='T':
    use_DAG=True
elif opt.use_DAG!='F':
    use_DAG=False
else:
    raise AssertionError('use_DAG should be T/F')
if use_DAG:
    train_data, val_data, test_data=preprocess_data_exp1_dag(pred_task,pred_tgt,pred_mode)
else:
    train_data, val_data, test_data=preprocess_data_exp1(pred_task,pred_tgt,pred_mode)

config=select_model_exp1()
if use_DAG==False:
    config.structure=False
train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=config.batch_size,num_workers=2,
                                        shuffle=False)
val_loader=torch.utils.data.DataLoader(dataset=val_data,batch_size=config.batch_size,num_workers=2,
                                        shuffle=False)
test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=config.batch_size,num_workers=2,
                                        shuffle=False)



if __name__=='__main__':
    model=DAGTransformer(config).to(config.device)
    train(config, model, train_loader, val_loader, test_loader)

    
