import torch
import torch.nn as nn 
import torch.optim as optim 
import os 
import sys
import numpy as np 
import cv2
import torchvision.transforms as transforms

from torch.utils.data import DataLoader,Dataset

import random
from tqdm import tqdm

import glob
reg_path='C:\\Users\\dongchan\\VScodeProjects\\GenderRecognition\\Dataset\\PersonReID Data\\RegDB'

    
import time    

  
def gender_labeling_reg(path):
    if (path.split("\\")[-1]).split("_")[0] == 'male':
        #return torch.tensor(0)
        return torch.tensor(0)
              
    elif (path.split("\\")[-1]).split("_")[0] == 'female':
        #return torch.tensor(1)
        return torch.tensor(1)
    
def gender_labeling_sysu(path):
    if (path.split("\\")[-1]).split("_")[0] == 'Male':
        return torch.tensor(0)
              
    elif (path.split("\\")[-1]).split("_")[0] == 'Female':
        return torch.tensor(1)
    
def split_validation_reg(paths):
    train_paths=paths.copy()
    persons=[]
    valid_paths=[]
    for i in paths:
        persons.append(i.split("\\")[-1].split("_")[-1].split(".")[0]) 
    persons = set(persons)
    persons = list(persons)
    random.seed(100)
    sample = random.sample(persons,int(len(persons)*0.1))
    for path in paths:
        j=path.split("\\")[-1].split("_")[-1].split(".")[0]
        if j in sample:
            valid_paths.append(path)
    for i in valid_paths:
        train_paths.remove(i)       
    return train_paths,valid_paths 

def split_validation_sysu(paths):
    train_paths=paths.copy()
    persons=[]
    valid_paths=[]
    for i in paths:
        persons.append(i.split("\\")[-1].split("_")[-2]) 
    persons = set(persons)
    persons = list(persons)
    random.seed(100)
    sample = random.sample(persons,int(len(persons)*0.1))
    for path in paths:
        j=path.split("\\")[-1].split("_")[-2]
        if j in sample:
            valid_paths.append(path)
    for i in valid_paths:
        train_paths.remove(i)       
    return train_paths,valid_paths 


class RegDB(Dataset):
    def __init__(self,root_path,transform,mode='train', K=1):
        super().__init__()
        print("\n---[ RegDB init ]---\n")
        self.root_path = root_path
        self.K = K 
        self.mode = mode
        self.train_paths=[]
        folds=[1,2,3,4,5]
        folds.remove(K)
        for i in folds:
            self.train_paths.extend(glob.glob(self.root_path+f"\\RegDB\\{i}\\*.bmp"))
        self.train_paths,self.valid_paths = split_validation_reg(self.train_paths)
        if self.mode =='train':  
           self.path = self.train_paths
        elif self.mode =='test':
            self.test_paths = glob.glob(self.root_path+f"\\RegDB\\{K}\\*.bmp")
            self.path = self.test_paths
        elif self.mode =='validation':
            self.path = self.valid_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
    
        img = cv2.imread(self.path[idx],0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        transformed = self.transform(image=img)
        image = transformed['image']
        label = gender_labeling_reg(self.path[idx])
        return image,label
    

class SYSU(Dataset):
    def __init__(self,root_path,transform,mode='train', K=1):
        super().__init__()
        print("\n---[ SYSU init ]---\n")
        self.root_path = root_path
        self.K = K 
        self.mode = mode
        self.train_paths=[]
        folds=[1,2]
        folds.remove(K)
        for i in folds:
            self.train_paths.extend(glob.glob(self.root_path+f"\\SYSU\\{i}\\*.jpg"))
        self.train_paths,self.valid_paths = split_validation_sysu(self.train_paths)
        if self.mode =='train':  
           self.path = self.train_paths
        elif self.mode =='test':
            self.test_paths = glob.glob(self.root_path+f"\\SYSU\\{K}\\*.jpg")
            self.path = self.test_paths
        elif self.mode =='validation':
            self.path = self.valid_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
    
        img = cv2.imread(self.path[idx],0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        transformed = self.transform(image=img)
        image = transformed['image']
        label = gender_labeling_sysu(self.path[idx])
        return image,label