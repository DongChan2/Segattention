import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TF
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from ray import tune
from tqdm import tqdm 
import time
from datetime import datetime
import os


from torch.utils.tensorboard import SummaryWriter
from mp_dataloader import DataLoader_multi_worker_FIX
import models

def fix_str(in_str):
    return "_".join(str(in_str).split(": "))

def save_checkpoint(epoch,model,optimizer,loss,name,scheduler=None):
    PATH= "D:/dongchan/Backup/data/"
    NEW_PATH=PATH+name
    os.makedirs(NEW_PATH,exist_ok=True)
    ckpt={"MODEL":model['classification'].state_dict(),
          "OPTIMIZER":optimizer.state_dict(),
          'EPOCH':epoch+1,
          "NAME":name}
    if scheduler is not None:
        ckpt.update({'SCHEDULER':scheduler,"SCHEDULER_STATE":scheduler.state_dict()})
    torch.save(ckpt,NEW_PATH+f"/Epoch_{epoch+1}.pth.tar")
    
    


def compute_eer(preds,targets):
    

    fpr, tpr, thresholds = roc_curve(targets, preds, pos_label=1,drop_intermediate=True)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
   
 
    return eer

def compute_accuracy(preds,target):
    ##preds.shape = N,C
    ##target.shape= N
    preds = torch.softmax(preds,dim=1)
    preds = (torch.argmax(preds,dim=1)).view(-1)
    #target=target.view(-1)
    target=target.view(-1)
    accuracy = ((preds==target).double().sum())/len(target)
    return accuracy
    

def _train_one_step(model,data,optimizer,device,**kwargs):
    logger = kwargs['logger']
    criterion = kwargs['criterion']

    x,y = data
    # x.requires_grad_(True)
    x=x.to(device)
    y=y.to(device)
    # print(x.requires_grad)
    # print(x.requires_grad)
    # loss = F.cross_entropy(model(x),y.long()) 
    #loss = criterion(model(x),y.long())
    #accuracy = compute_accuracy(model(x),y.long())
    with torch.no_grad():
        infer = torch.nn.functional.softmax(model['segmentation'](x),dim=1)
        s=infer[:,1,:,:].unsqueeze(1)
    positive = s*x
    negative = (1.-s)*x
    pred,anchor,positive,negative = model['classification'](x,positive,negative)
    
    
    optimizer.zero_grad()
    loss =  criterion['CE'](pred,y) + 0.5*criterion['TRIPLET'](anchor,positive,negative)
    accuracy = compute_accuracy(pred,y)
    
    logger.add_scalar("loss/step",loss,kwargs['iter'])
    logger.add_scalar("accuracy/step",accuracy,kwargs['iter'])
    
    loss.backward()
    optimizer.step()
    
    return {'loss':loss.item(),'accuracy':accuracy.item()}
    

def _train_one_epoch(model,dataloader,optimizer,device,**kwargs):
    model['segmentation'].eval()
    model['classification'].train()
    
    total_loss = 0
    total_accuracy = 0
    
    for batch_index,data in enumerate(tqdm(dataloader)):
        
        history = _train_one_step(model,data,optimizer,device,logger=kwargs['logger'],iter=(len(dataloader)*(kwargs['epoch_index'])+(batch_index+1)), criterion = kwargs['criterion'])
        total_loss += history['loss']
        total_accuracy += history['accuracy']

    return {'loss':total_loss,'accuracy':total_accuracy}



def _validate_one_step(model,data,device,*args,**kwargs):
    logger = kwargs['logger']
    criterion = kwargs['criterion']
    x,y = data
    x=x.to(device)
    y=y.to(device)
    # loss = F.cross_entropy(model(x),y.long())
    #loss = criterion(model(x),y.long())
    #accuracy = compute_accuracy(model(x),y.long())
    with torch.no_grad():
        infer = torch.nn.functional.softmax(model['segmentation'](x),dim=1)
        s=infer[:,1,:,:].unsqueeze(1)
    positive = s*x
    negative = (1.-s)*x
    pred,anchor,positive,negative = model['classification'](x,positive,negative)
    
    
    loss = criterion['CE'](pred,y)+0.5*criterion['TRIPLET'](anchor,positive,negative)
    accuracy = compute_accuracy(pred,y)
   
    logger.add_scalar("loss/step",loss,kwargs['iter'])
    logger.add_scalar("accuracy/step",accuracy,kwargs['iter'])
    
    return {'loss':loss.item(),'accuracy':accuracy.item()}
    

def _validate_one_epoch(model,dataloader,device,**kwargs):

    for i in model.values():
        i.eval()
    total_loss = 0
    total_accuracy = 0
    for batch_index,data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            history = _validate_one_step(model,data,device,logger=kwargs['logger'],iter=(len(dataloader)*(kwargs['epoch_index'])+(batch_index+1)), criterion = kwargs['criterion'])
        total_loss += history['loss']
        total_accuracy += history['accuracy']
        
    
    return {'loss':total_loss,'accuracy':total_accuracy}



def _test_one_step(model,data,device,*args,**kwargs):
    
    x,y=data
    x=x.to(device)
    y=y.to(device)
    
    pred,a,b,c=model(x,x,x)
    return pred,y
  
    

def _test_one_epoch(model,dataloader,device,*args,**kwargs):
    model=model['classification']
    criterion=kwargs['criterion']
    total_loss =0
    total_acc=0
    total_pred=torch.tensor([],device=device)
    total_y=torch.tensor([],device=device)


    for data in tqdm(dataloader):
        with torch.no_grad():
            pred,y = _test_one_step(model,data,device)

        total_loss+=criterion['CE'](pred,y)
        total_acc+=compute_accuracy(pred,y)
        total_pred=torch.cat([total_pred,torch.softmax(pred,dim=1)[:,1].view(-1)],dim=0)
        total_y=torch.cat([total_y,y.view(-1)],dim=0)
    eer = compute_eer(total_pred.detach().cpu(),total_y.detach().cpu())
    acc = total_acc/len(dataloader)
    loss = total_loss/len(dataloader)

    result = {'eer':eer,'loss':loss,'acc':acc}
    return result 

def train(model,loaders,optimizer,hparam,device,lr_scheduler=None,save_ckpt=True):
    dataloader,valid_dataloader,test_dataloder = loaders
    print("Training Start")
    print("="*100)
    t= datetime.today().strftime("%m_%d")
    
    _path = "./logs/" + t
    
    _count = 1
    _path_sub = "/ver_"
    _path_out = _path + _path_sub + str(_count)
    while os.path.exists(_path_out):
        _count += 1
        _path_out = _path + _path_sub + str(_count)
    
    
    if hparam["K"] == 1:
        name = t + _path_sub + str(_count) + "/" + fix_str(str(hparam))
    else:
        name = t + _path_sub + str(_count - 1) + "/" + fix_str(str(hparam))
    
    #t="layer2,layer3,layer4,fc"
    train_logger = SummaryWriter(log_dir = f"./logs/{name}/train")
    
    valid_logger = SummaryWriter(log_dir = f"./logs/{name}/validation")
    test_logger = SummaryWriter(log_dir = f"./logs/{name}/test")

    
    model['segmentation'].load_state_dict(models.seg_state_dict)
    model['segmentation'].to(device)
    
    model['classification'].to(device)
    
    epochs = hparam['EPOCH']
    
    criterion = {'CE':nn.CrossEntropyLoss(),"TRIPLET":nn.TripletMarginLoss()}
    
    for idx,epoch in (enumerate(range(epochs))):
        
        print(f"\rEpoch :{idx+1}/{epochs}")
        history = _train_one_epoch(model,dataloader,optimizer,device,epoch_index=idx,logger=train_logger,criterion=criterion)
        epoch_loss = history['loss'] / len(dataloader)
        epoch_accuracy = history['accuracy'] / len(dataloader)
        
        train_logger.add_scalar("loss/epoch",epoch_loss,idx+1)
        train_logger.add_scalar("accuracy/epoch",epoch_accuracy,idx+1)
        val_history = _validate_one_epoch(model,valid_dataloader,device,epoch_index=idx,logger=valid_logger,criterion=criterion)
        epoch_val_loss = val_history['loss'] / len(valid_dataloader)
        epoch_val_accuracy = val_history['accuracy'] / len(valid_dataloader)
        valid_logger.add_scalar("loss/epoch",epoch_val_loss,idx+1)
        valid_logger.add_scalar("accuracy/epoch",epoch_val_accuracy,idx+1)
        
        test_history = _test_one_epoch(model,test_dataloder,device,criterion=criterion)
        test_logger.add_scalar("loss/epoch",test_history['loss'],epoch+1)
        test_logger.add_scalar("result/eer",test_history['eer'],idx+1)
        test_logger.add_scalar("accuracy/epoch",test_history['acc'],idx+1)
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        if save_ckpt:
            save_checkpoint(epoch,model,optimizer,epoch_loss,name=name,scheduler=lr_scheduler)
        print(f"loss:{epoch_loss},acc:{epoch_accuracy},valid_loss:{epoch_val_loss},valid_acc:{epoch_val_accuracy}")
        print(test_history)
    train_logger.close()
    valid_logger.close()
    test_logger.close()
    print('Training End')
    print("="*100)



def test(model,dataloader,hparam,device,ckpt_path):
    print("Testing Start")
    print("="*20)
    ckpt = torch.load(ckpt_path)
    ckpt_model = ckpt['MODEL']
    for m,c in zip(model,ckpt_model):
        m.load_state_dict(c)
        m.to(device)
    criterion = nn.CrossEntropyLoss()
    test_history = _test_one_epoch(model,dataloader,device,criterion=criterion)
    print(test_history)
    print("Testing End")
    print("="*20)
    
def resume_train(model,dataloader,valid_dataloader,optimizer,hparam,device,ckpt_path:str,lr_scheduler=None,save_ckpt=True):
    print("Resume Training")
    print("="*100)
    ckpt = torch.load(ckpt_path)
    ckpt_model = ckpt['MODEL']
    name = ckpt['NAME']
    for m,c in zip(model,ckpt_model):
        m.load_state_dict(c)
        m.to(device)
    
    optimizer.load_state_dict(ckpt['OPTIMIZER'])
    if 'SCHEDULER' in ckpt.keys():
        lr_scheduler=ckpt['SCHEDULER']
        lr_scheduler.load_state_dict(ckpt['SCHEDULER_STATE'])
        for state in lr_scheduler.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    if device == 'cuda':
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    
    start_epoch = ckpt['EPOCH']
    end_epoch = hparam['EPOCH']
    
    
    train_logger = SummaryWriter(log_dir = f"./logs/{name}/train")
    valid_logger = SummaryWriter(log_dir = f"./logs/{name}/validation")

   
   
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in ((range(start_epoch,end_epoch))):
        
        print(f"\rEpoch :{epoch+1}/{end_epoch}")
        print("="*100)
        #@@ Training Phase
        history = _train_one_epoch(model,dataloader,optimizer,device,epoch_index=epoch,logger=train_logger,criterion=criterion)
        epoch_loss = history['loss'] / len(dataloader)
        epoch_accuracy = history['accuracy'] / len(dataloader)
        train_logger.add_scalar("loss/epoch",epoch_loss,epoch+1)
        train_logger.add_scalar("accuracy/epoch",epoch_accuracy,epoch+1)
        
        #@@ Validation Phase
        val_history = _validate_one_epoch(model,valid_dataloader,device,epoch_index=epoch,logger=valid_logger,criterion=criterion)
        epoch_val_loss = val_history['loss'] / len(valid_dataloader)
        epoch_val_accuracy = val_history['accuracy'] / len(valid_dataloader)
        valid_logger.add_scalar("loss/epoch",epoch_val_loss,epoch+1)
        valid_logger.add_scalar("accuracy/epoch",epoch_val_accuracy,epoch+1)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if save_ckpt:
            save_checkpoint(epoch,model,optimizer,epoch_loss,name=name,scheduler=lr_scheduler)
    train_logger.close()
    valid_logger.close()
    print("Resume End")
    print("="*100)



if __name__ == '__main__':
    pass