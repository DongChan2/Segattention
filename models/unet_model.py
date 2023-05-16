""" Full assembly of the parts to form the complete network """
import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .unet_parts import *
import pytorch_lightning as pl
from typing import Any
import torch.nn as nn
import os
from torchmetrics.classification import BinaryJaccardIndex,BinaryAccuracy
import numpy as np


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

        
       

    def training_step(self,batch, batch_idx):   #@@ (필수 적으로 정의)
        x,y = batch
        y_hat = self(x).squeeze_(1)
        
        y = torch.where(y>0, 1., 0.)
        loss = nn.BCEWithLogitsLoss()(y_hat,y)
        acc = BinaryAccuracy().to(self.device)(y_hat,y)
        iou = BinaryJaccardIndex().to(self.device)(y_hat,y)
        self.log_dict({"train/loss": loss,
                          "train/iou": iou,
                          "train/acc": acc},on_epoch = True,on_step=True, prog_bar=True)
        # and the average across the epoch, to the progress bar and logger

        return loss

   

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x).squeeze_(1)
        y = torch.where(y>0, 1., 0)
        cls_loss = nn.BCEWithLogitsLoss()(y_hat, y)
  
        loss = cls_loss 
      
        acc = BinaryAccuracy().to(self.device)(y_hat,y)
        iou = BinaryJaccardIndex().to(self.device)(y_hat,y)
        self.log_dict({"val/loss": loss,
                          "val/iou": iou,
                          "val/acc": acc},on_epoch = True,on_step=True, prog_bar=True)
        return loss

    def inference(self,x:np.array)->np.array:
        
        x=A.Compose([
            A.ToFloat(),
                     
                     ToTensorV2()
                     
                     ])(image=x)
        x=x['image'].unsqueeze_(0)
        self.eval()

        with torch.no_grad():
            x=x.to(self.device)
            x=self(x)[0]
            head_x = self(x)[1:]

        x.squeeze_(0).squeeze_(0)
        x=torch.sigmoid(x)
        head_x=[torch.sigmoid(i) for i in head_x]
        x=x.detach().cpu().numpy()
        return x,head_x  
    
            
    def configure_optimizers(self) -> Any:
        optimizer= torch.optim.AdamW(self.parameters(), lr=1e-4,capturable=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,patience=5,factor=0.1)
        
        return {'optimizer': optimizer,'lr_scheduler': lr_scheduler,'monitor':'val/loss' }
    
class UNet_pl_ver2(pl.LightningModule):
    def __init__(self, n_channels, n_classes,memo="", bilinear=False, learning_rates = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['bilinear'])
        self.logs =dict()
        self.base_unet = UNet(1,1, bilinear=False)
        self.encoder = torch.mean
        


        self.training_step_outputs = []
    def forward(self, x):
        x = self.base_unet(x)
        head_x = self.encoder(torch.sigmoid(x[:,:,:int(x.shape[2]*0.15),:]),dim=(1,2,3)).view(-1,1)
        upper_x = self.encoder(torch.sigmoid(x[:,:,int(x.shape[2]*0.15):int(x.shape[2]*0.6),:]),dim=(1,2,3)).view(-1,1)
        lower_x = self.encoder(torch.sigmoid(x[:,:,int(x.shape[2]*0.6):,:]),dim=(1,2,3)).view(-1,1)
        return [x, head_x, upper_x, lower_x]
        
        
       

    def training_step(self,batch, batch_idx):   #@@ (필수 적으로 정의)
        x,y = batch
        y_hat = self(x)[0].squeeze_(1)
        y.squeeze_(1)
        y = torch.where(y>0, 1., 0)
        cls_loss = nn.BCEWithLogitsLoss()(y_hat,y)
        reg_loss = RegLoss()(logits=self(x)[1:],targets=y)
        loss = cls_loss 
        acc = BinaryAccuracy().to(self.device)(y_hat,y)
        iou = BinaryJaccardIndex().to(self.device)(y_hat,y)
        self.logs.update({"train_loss": loss,
                          "train_iou": iou,
                          "train_acc": acc})
        # and the average across the epoch, to the progress bar and logger

        return loss

    def on_train_epoch_end(self):           #@@ (Optional) 1 epoch이 끝난 뒤 모든 train prediction값들을 이용한 연산이 필요할 경우 정의

        pass
        #all_preds = torch.stack(self.training_step_outputs)
        #@@ ... train 과정에서 생성된 모든 prediction값을 이용한 연산 정의
        #self.training_step_outputs.clear()  # free memory

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)[0].squeeze_(1)
        y.squeeze_(1)
        y = torch.where(y>0, 1., 0)
        cls_loss = nn.BCEWithLogitsLoss()(y_hat, y)
        reg_loss = RegLoss()(logits=self(x)[1:],targets=y)
        loss = cls_loss 
      
        acc = BinaryAccuracy().to(self.device)(y_hat,y)
        iou = BinaryJaccardIndex().to(self.device)(y_hat,y)
        self.logs.update({"val_loss": loss,
                          "val_iou": iou,
                          "val_acc": acc})
        self.log_dict(self.logs,on_epoch = True, logger = True, prog_bar=True)
        return loss

    def inference(self,x:np.array)->np.array:
        
        x=A.Compose([A.ToFloat(),
                     
                     ToTensorV2()
                     
                     ])(image=x)
        x=x['image'].unsqueeze_(0)
        self.eval()

        with torch.no_grad():
            x=x.to(self.device)
            x=self(x)[0]
            head_x = self(x)[1:]

        x.squeeze_(0).squeeze_(0)
        x=torch.sigmoid(x)
        
        x=x.detach().cpu().numpy()
        return x,head_x
        
    def on_validation_epoch_end(self) -> None:
        self.logs.clear()
        
    
            
                




    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), lr=1e-4,capturable=True)
    
    
    
    
class UNet_AutoEncoder(pl.LightningModule):
    def __init__(self, n_channels, n_classes,memo="", bilinear=False, learning_rates = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['bilinear'])
        self.logs =dict()
        self.base_unet = UNet(1,1, bilinear=False)
  
        


        self.training_step_outputs = []
    def forward(self, x):
        x = self.base_unet(x)
        return x
        
       

    def training_step(self,batch, batch_idx):   #@@ (필수 적으로 정의)
        x,y = batch
        y_hat = self(x)
 
        loss = nn.MSELoss()(y_hat,y)
 
     
        self.logs.update({"train_loss": loss,
                          "train_iou": iou,
                          "train_acc": acc})
        # and the average across the epoch, to the progress bar and logger

        return loss

    def on_train_epoch_end(self):           #@@ (Optional) 1 epoch이 끝난 뒤 모든 train prediction값들을 이용한 연산이 필요할 경우 정의

        pass
        #all_preds = torch.stack(self.training_step_outputs)
        #@@ ... train 과정에서 생성된 모든 prediction값을 이용한 연산 정의
        #self.training_step_outputs.clear()  # free memory

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)[0].squeeze_(1)
        y.squeeze_(1)
        y = torch.where(y>0, 1., 0)
        cls_loss = nn.BCEWithLogitsLoss()(y_hat, y)
        reg_loss = RegLoss()(logits=self(x)[1:],targets=y)
        loss = cls_loss + reg_loss
        acc = BinaryAccuracy().to(self.device)(y_hat,y)
        iou = BinaryJaccardIndex().to(self.device)(y_hat,y)
        self.logs.update({"val_loss": loss,
                          "val_iou": iou,
                          "val_acc": acc})
        self.log_dict(self.logs,on_epoch = True, logger = True, prog_bar=True)
        return loss

    def inference(self,x:np.array)->np.array:
        mean = (0.6023180499509141,)
        std = (0.06016134033226093,)
        x=T.Compose([T.ToTensor(),
                     T.Normalize(mean, std)])(x)
        x.unsqueeze_(0)
        self.eval()
        with torch.no_grad():
            x=x.to(self.device)
            x=self(x)[0]
            head_x = self(x)[1:]

        x.squeeze_(0).squeeze_(0)
        x=torch.sigmoid(x)
        head_x=[torch.sigmoid(i) for i in head_x]
        x=x.detach().cpu().numpy()
        return x,head_x

    def on_validation_epoch_end(self) -> None:
        self.logs.clear()




    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


