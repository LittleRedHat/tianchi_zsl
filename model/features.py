import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from .modules.mobilenetv2 import MobileNetV2



class MobileNet(nn.Module):
    def __init__(self,num_classes,input_size):
        super(MobileNet,self).__init__()
        self.model = MobileNetV2(n_class=num_classes,input_size=input_size)

    def forward(self,x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        logits = self.classifier(x)
        return logits,x

class SENet(nn.Module):
    pass

class Resnet(nn.Module):
    def __init__(self,num_classes):
        super(Resnet,self).__init__()

    
    def forward(self,x)



    






