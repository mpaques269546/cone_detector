import torch
from torch import nn
from torchvision import datasets,  transforms
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

class MyTransform:
    def __init__(self,  hflip=False, vflip=False, rcrop=False, ccrop=False, colorjitter=False, size=None, normalize=True):
        self.size = (size, size)
        self.max_size = 5000
        self.hflip = hflip #random horizontal flip
        self.vflip = vflip
        self.rcrop = rcrop # random crop
        self.ccrop = ccrop # central crop
        self.colorjitter = colorjitter# random colorjitter
        self.color_transform = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], # hue=0.1
            p=0.8) 
        self.normalize = normalize
    

    def __call__(self, x, y):
        if not torch.is_tensor(x):
            x = TF.to_tensor(x)
        if not torch.is_tensor(y):
            y = TF.to_tensor(y)
        
        assert x.shape[-2:]==y.shape[-2:]
        
        if self.size[0]!=None:
            if self.rcrop:
                if x.shape[-2] > self.size[0] and x.shape[-1] > self.size[1]:
                    # resize first to max size
                    #x = TF.resize(x, self.max_size) # original
                    #y = TF.resize(y, self.max_size)
                    
                    top = random.randint(0, x.shape[-2] - self.size[0])
                    left =  random.randint(0, x.shape[-1] - self.size[1] )
                    x = TF.crop(x , top, left, height=self.size[0], width=self.size[1])
                    y = TF.crop(y , top, left, height=self.size[0], width=self.size[1])
                else:
                    x = TF.resize(x, self.size) # original
                    y = TF.resize(y, self.size)
            elif self.ccrop:
                x = transforms.CenterCrop(self.size)(x)
                y = transforms.CenterCrop(self.size)(y)

            else:
                x = TF.resize(x, self.size) # original
                y = TF.resize(y, self.size)
                

        if self.colorjitter:
            x = self.color_transform(x)
            

        if self.hflip:
            rand = random.randint(0,1)
            if rand==1:
                x = TF.hflip(x)
                y = TF.hflip(y)

        if self.vflip:
            rand = random.randint(0,1)
            if rand==1:
                x = TF.vflip(x)
                y = TF.vflip(y)

        if self.normalize:
            x = TF.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return x, y



class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Args:
            image_path (string): Path to the images.
            targets (pd df): labels 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path
        self.images_path = data_path + 'images/'
        self.masks_path = data_path + 'masks/'
        self.class_list = ['background', 'cone']
        self.num_cls = len(self.class_list)
        self.images_list = os.listdir(self.images_path) 
        self.masks_list = os.listdir(self.masks_path)
        self.mtype = self.masks_list[0][self.masks_list[0].index('.'):]
        self.itype = self.images_list[0][self.images_list[0].index('.'):]
        self.transform = transform
        
       
        
    def __len__(self):
        return len(self.masks_list)
            

    def  __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx)==list:
            idx = idx[0]
        mask_name = self.masks_list[idx]
        mask_path = os.path.join( self.masks_path, mask_name)

        img_name = mask_name.replace("_cone.ome.tiff", "").replace(".jpg", "").replace("_tree.ome.tiff", '') +'.jpg'  #mask_name.split('_')[0]+'.jpg'
        img_path = os.path.join( self.images_path, img_name)
                
        if os.path.isfile(img_path):
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        else:
            print(f'ERROR: missing img {img_path}')

        if os.path.isfile(mask_path):
            with open(mask_path, 'rb') as f:
                mask = Image.open(f)
                mask = mask.convert('L') #.convert('RGB')
                
        else:
            print(f'ERROR: missing mask {mask_path}')
        
        if self.transform:
            img, mask = self.transform(img, mask)
        
        c,h,w = img.shape
        target = torch.zeros(self.num_cls, h, w)
        mask = (mask[0]>0)*1
        for i in range(self.num_cls):
            target[i]= (mask==i)*1
        return img, target.float() #.unsqueeze(0).long() # [b,3,h,w], [b,n_cls,h,w]

