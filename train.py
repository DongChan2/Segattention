import torch
from torch.optim import *
from argparse import ArgumentParser
import dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np 

from models.resnet_model import *
from mp_dataloader import DataLoader_multi_worker_FIX
from torch.utils.data import DataLoader
import trainer
import models
def main(args,k):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    hparam ={'EPOCH':args.epoch,'BATCH_SIZE':args.batch_size,'lr':args.lr,'weight_decay':args.weight_decay,"K":k,"backbone":args.backbone,'DS':args.dataset}
    
    
    root_path = "C:\\Users\\dongchan\\VScodeProjects\\GenderRecognition\\Dataset"
    seg_ckpt_path=f".\\models\\unet_weights\\{args.dataset}\\{k}\\weight.pth.tar"

    seg_ckpt = torch.load(seg_ckpt_path)['MODEL']
    transform = {'origin': 
                            A.Compose([
                            A.Resize(384,128),
                            A.ToFloat(),
                            A.HorizontalFlip(p=0.5),
                            A.OneOf([A.Affine(translate_px={'x':(-30,30),'y':(-10,20)},keep_ratio=True,p=0.5),
                                    A.Affine(rotate=20,keep_ratio=True,p=0.5)],p=0.5),
                            A.RandomBrightnessContrast(brightness_limit=(-0.3,0.1),contrast_limit=(-0.2,0.3),p=0.3),
                            ToTensorV2()]),
                                          
                  "valid": A.Compose([
                            A.Resize(384,128),
                            A.ToFloat(),
                    
                            ToTensorV2()
                            ] 
                                    )
                }
    #@@ RegDB dataset
    train_ds = getattr(dataset,args.dataset)(root_path=root_path, transform=transform['origin'], K=k,mode='train')
    valid_ds = getattr(dataset,args.dataset)(root_path=root_path, transform=transform['valid'], K=k,mode='validation')
    test_ds = getattr(dataset,args.dataset)(root_path=root_path, transform=transform['valid'], K=k,mode='test')
    
    #@@ SYSU dataset
    # train_ds = SYSU(root_path=root_path, transform=transform['origin'], K=k,mode='train')
    # valid_ds = SYSU(root_path=root_path, transform=transform['valid'], K=k,mode='validation')
    # test_ds = SYSU(root_path=root_path, transform=transform['valid'], K=k,mode='test')
    
    train_loader = DataLoader(dataset=train_ds,batch_size=hparam['BATCH_SIZE'],pin_memory=True, shuffle= True,num_workers=0)
    valid_loader = DataLoader(dataset=valid_ds,batch_size=32,pin_memory=True, shuffle= False,num_workers=0)
    test_loader = DataLoader(dataset=test_ds,batch_size=32,pin_memory=True, shuffle= False,num_workers=0)
    
    loaders=[train_loader,valid_loader,test_loader]
    # model_path = f"D:\\dongchan\\Backup\\data\\" + args.model_path
    
    model = {"classification":models.Classification_model(name=hparam['backbone']),"segmentation":models.UNet(3,2)}
    model['segmentation'].load_state_dict(seg_ckpt)
    
    optimizer = AdamW([{'params':model['classification'].parameters()}],lr=hparam['lr'],weight_decay=hparam['weight_decay'])

    # optimizer = Adam(model['classification'].parameters(),lr=hparam['lr'],weight_decay=hparam['weight_decay'])
    # optimizer = SGD(model['classification'].parameters(),lr=hparam['lr'],momentum=0.9,weight_decay=hparam['weight_decay'])

    #   lr_scheduler =None
    # lr_scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=3)
    lr_scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=hparam['EPOCH'],eta_min=1e-6)
   
  
    # CKPT_PATH="D:/dongchan/Backup/data/GenderClassification/resnet101/2023_04_09 18_20_02/Epoch_0.pth.tar"
    trainer.train(model,loaders,optimizer,hparam,device,lr_scheduler=lr_scheduler,save_ckpt=True)
    # trainer.resume_train(model,train_loader,valid_loader,optimizer,hparam,device,ckpt_path=CKPT_PATH)
    # trainer.test(model,test_loader,hparam,device,ckpt_path=CKPT_PATH)

if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=16,type = int)
    parser.add_argument("--epoch", default=100,type = int)
    parser.add_argument("--lr", default=1e-4,type = float)
    parser.add_argument("--weight_decay", default=1e-4,type = float)
    parser.add_argument("--backbone", default='resnet101',type = str)
    parser.add_argument("--dataset", default='SYSU',type = str)
    args = parser.parse_args()
    if args.dataset == 'RegDB':
        for i in range(1,6):
            main(args,k=i)
    elif args.dataset == 'SYSU': 
        for i in range(1,3):
            main(args,k=i)
        

    