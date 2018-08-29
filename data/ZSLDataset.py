import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os

class ZSLDataset(Dataset):
    def __init__(self,image_dir,sample_file,embed_file,attribute_file,lable_name_file,transformer=None):
        super(ZSLDataset,self).__init__()

        self.transformer = transformer
        self.image_dir = image_dir

        label_to_name = {}
        label_to_idx = {}
        idx = 0
        with open(lable_name_file,'r') as f:
            for line in f:
                label,name = line.strip().split()
                label_to_name[label] = name

                if not label in label_to_idx:
                    label_to_idx[label] = idx
                    idx = idx + 1
        self.label_to_name = label_to_name
        self.label_to_idx = label_to_idx

        
        
        label_to_attribute = {}
        with open(attribute_file,'r') as f:
            for line in f:
                label_attribute = line.strip().split()
                label = label_attribute[0]
                attribute = [float(_) for _ in label_attribute[1:]]
                label_to_attribute[label] = attribute
        self.label_to_attribute = label_to_attribute

        name_to_embed = {}
        with open(embed_file,'r') as f:
            for line in f:
                name_embed = line.strip().split()
                name = name_embed[0]
                embed = [float(_) for _ in name_embed[1:]]
                name_to_embed[name] = embed
        self.name_to_embed = name_to_embed

        images = []
        labels = []
        with open(sample_file,'r') as f:
            for line in f:
                name,label = line.strip().split()
                images.append(name)
                labels.append(self.label_to_idx(label))

        self.images = images
        self.labels = labels
        

    
    def image_preprocess(self,image_path,transformer):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = transformer(image)
        return image

    def get_class_size(self):
        return len(self.label_to_idx)
    
    def __getitem__(self,index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir,image_name)

        image = self.image_preprocess(image_path,transformer=self.transformer)
        label = self.labels[index]
        embed = self.name_to_embed[self.label_to_name(label)]
        attribute = self.label_to_attribute(label)

        return image,\
               torch.tensor(label,type=torch.long),\
               torch.tensor(embed,type=torch.float),\
               torch.tensor(attribute,type=torch.float)


    def __len__(self):
        return len(self.samples)

    def get_dataloader(self,batch_size=32,num_workers=0,shuffle=True):
        dataloader = DataLoader(self,batch_size=batch_size,num_workers=num_workers,shuffle=shuffle)
        return dataloader


    def collate_fn(self,sequences):
        pass
