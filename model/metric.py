import torch 
import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score



def compute_acc(preds,targets):
    preds = np.argmax(preds,axis=1)
    acc = accuracy_score(targets,preds)
    return acc



