import torch 
import torch.nn as nn
import numpy as np 
from sklearn.metrics import log_loss


def compute_cross_loss(preds,targets):
    loss = log_loss(targets,preds)
    return loss