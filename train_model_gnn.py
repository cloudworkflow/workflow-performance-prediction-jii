from tqdm import tqdm
from scheduler import WarmUpLR, downLR
import time
import torch
import numpy as np
import torch.nn.functional as F
from datetime import timedelta
from sklearn import metrics
import time
from datetime import timedelta
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(config, model, data):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    warmup_epoch = config.num_epochs/2
    scheduler=downLR(optimizer, (config.num_epochs-warmup_epoch))
    
    warmup_scheduler = WarmUpLR(optimizer, warmup_epoch)
    total_batch = 0  
    dev_best_loss = float('inf')
    dev_best_acc = float(0)
    test_best_acc=float(0)
    lrlist=np.zeros((config.num_epochs,2))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        lrlist[epoch][0]=epoch
        if(epoch>=warmup_epoch):
            
            learn_rate = scheduler.get_lr()[0]
            print("Learn_rate:%s" % learn_rate)
            lrlist[epoch][1]=learn_rate
        else:
            learn_rate = warmup_scheduler.get_lr()[0]
            lrlist[epoch][0]=learn_rate
            print("Learn_rate:%s" % learn_rate)
        
        
        data=data.to(config.device)
        outputs = model(data)
        model.zero_grad()
        loss = F.cross_entropy(outputs[data.train_mask], data.labels[data.train_mask])   
        loss.backward()
        optimizer.step()
        if(epoch<warmup_epoch):
            warmup_scheduler.step()
        else:
            scheduler.step()
        total_batch += 1
        pred_tr=torch.max(outputs[data.train_mask], 1)[1]
        true_tr=data.labels[data.train_mask]
        train_acc  = get_accuracy(pred_tr, true_tr)
        lossoutput=loss
                
        
        dev_acc, dev_loss = evaluate(config, model, data)
        test_acc, test_loss = test(config, model, data)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            
            improve = '*'
                    
        else:
            improve = ''
        if dev_acc > dev_best_acc:
            dev_best_acc = dev_acc
            #torch.save(model.state_dict(), './best1nod.ckpt')
            test_best_acc = test_acc
            
        time_dif = get_time_dif(start_time)
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Test Loss: {5:>5.2}, Test Acc: {6:>6.2%},Time: {7} {8}'
        print(msg.format(total_batch, lossoutput.item(), train_acc, dev_loss, dev_acc, test_loss,test_acc,time_dif, improve))
        model.train()
        print('BEST SO FAR:')
        print('Val Best Acc:', dev_best_acc)
        print('Test Best Acc:', test_best_acc)
    test(config, model, data, final=True)


def test(config, model, data, final=False):
    # test
    
    model.eval()
    with torch.no_grad():
        outputs=model(data)
        loss_total = F.cross_entropy(outputs[data.test_mask], data.labels[data.test_mask])   
        predict_all=torch.max(outputs[data.test_mask], 1)[1]
        labels_all=data.labels[data.test_mask]
    acc = get_accuracy(labels_all, predict_all)
    if final:
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(loss_total, acc))
        print("Confusion Matrix...")
        confusion = metrics.confusion_matrix(labels_all.cpu().numpy(), predict_all.cpu().numpy())
        print(confusion)
        return acc, loss_total, confusion
    return acc, loss_total
    


def evaluate(config, model, data):
    model.eval()
    with torch.no_grad():
        
        outputs=model(data)
        loss_total = F.cross_entropy(outputs[data.val_mask], data.labels[data.val_mask])   
        predict_all=torch.max(outputs[data.val_mask], 1)[1]
        labels_all=data.labels[data.val_mask]
    
    acc = get_accuracy(labels_all, predict_all)
    
    return acc, loss_total
from sklearn import metrics
def get_accuracy(y_true,y_pred):
    y_true,y_pred=y_true.cpu().numpy(), y_pred.cpu().numpy()
    return metrics.accuracy_score(y_true,y_pred)

