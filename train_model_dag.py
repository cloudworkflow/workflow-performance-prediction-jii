from tqdm import tqdm
from scheduler import WarmUpLR, downLR
import time
import torch
import numpy as np
import torch.nn.functional as F
from datetime import timedelta
from sklearn import metrics

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    warmup_epoch = config.num_epochs/2
    iter_per_epoch = len(train_iter)
    scheduler=downLR(optimizer, (config.num_epochs-warmup_epoch)*iter_per_epoch)
    
    warmup_scheduler = WarmUpLR(optimizer, warmup_epoch*iter_per_epoch)
    total_batch = 0  
    dev_best_loss = float('inf')
    dev_best_acc = float(0)
    test_best_acc=float(0)
    
    lrlist=np.zeros((config.num_epochs,2))
    for epoch in range(config.num_epochs):
        
        
        loss_total = 0
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        lrlist[epoch][0]=epoch
        predic_all = torch.LongTensor([]).to(config.device)
        true_all = torch.LongTensor([]).to(config.device)
        if(epoch>=warmup_epoch):
            
            learn_rate = scheduler.get_lr()[0]
            print("Learn_rate:%s" % learn_rate)
            lrlist[epoch][1]=learn_rate
        else:
            learn_rate = warmup_scheduler.get_lr()[0]
            lrlist[epoch][0]=learn_rate
            print("Learn_rate:%s" % learn_rate)
        
        
        for  (trains, labels, poss, masks) in tqdm(train_iter):
            trains=trains.to(config.device)
            #trains.dtype=torch.float
            labels=labels.long().to(config.device)
            #print(labels.dtype)
            poss=poss.to(config.device)
            masks=masks.to(config.device)
            outputs = model(trains,poss,masks)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
           
            
            optimizer.step()
            if(epoch<warmup_epoch):
                warmup_scheduler.step()
            else:
                scheduler.step()
            total_batch += 1
            loss_total+=loss
            
            true = labels.data
            predic = torch.max(outputs.data, 1)[1]
            predic_all=torch.cat((predic_all, predic),0)
            true_all=torch.cat((true_all, true),0)
            
        train_acc  = get_accuracy(true_all, predic_all)
        lossoutput=loss_total/len(train_iter)
                
                
        dev_acc, dev_loss = evaluate(config, model, dev_iter)
        
        
        test_acc, test_loss = evaluate(config, model, test_iter)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            
            improve = '*'
        else:
            improve = ''
        if dev_acc > dev_best_acc:
            dev_best_acc = dev_acc
            #torch.save(model.state_dict(), './best.ckpt')
            test_best_acc = test_acc
        time_dif = get_time_dif(start_time)
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Test Loss: {5:>5.2}, Test Acc: {6:>6.2%},Time: {7} {8}'
        print(msg.format(total_batch, lossoutput.item(), train_acc, dev_loss, dev_acc, test_loss,test_acc,time_dif, improve))
        print('BEST SO FAR:')
        print('Val Best Acc:', dev_best_acc)
        print('Test Best Acc:', test_best_acc)
        model.train()
    test(config, model, test_iter)


def test(config, model, test_iter):
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = torch.LongTensor([]).to(config.device)
    labels_all = torch.LongTensor([]).to(config.device)
    with torch.no_grad():
        for texts, labels, poss, masks in data_iter:
            texts=texts.float().to(config.device)
            poss=poss.float().to(config.device)
            masks=masks.float().to(config.device)
            labels=labels.long().to(config.device)
            outputs = model(texts,poss,masks)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data
            predic = torch.max(outputs.data, 1)[1]
            labels_all = torch.cat((labels_all, labels),0)
            predict_all = torch.cat((predict_all, predic),0)
    
    acc = get_accuracy(labels_all, predict_all)
    if test:
        confusion = metrics.confusion_matrix(labels_all.cpu().numpy(), predict_all.cpu().numpy())
        return acc, loss_total / len(data_iter), confusion
    return acc, loss_total / len(data_iter)

def get_accuracy(y_true,y_pred):
    y_true,y_pred=y_true.cpu().numpy(), y_pred.cpu().numpy()
    return metrics.accuracy_score(y_true,y_pred)