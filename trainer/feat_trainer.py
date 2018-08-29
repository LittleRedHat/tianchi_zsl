import torch
import torch.nn as nn

class FeatTrainer:
    def __init__(self,backbone='mobilenet',train_logger=None):
        
    

    def _build_model(self,bakcbone,num_classes):
        pass

    
    def _build_optimizer(self):
        pass





    def train(self):
        
        for epoch in range(self.start_epoch,self.epoches + 1):
            result = self._train_epoch()

    def _train_epoch(self):
        pass
    

    def _resume_checkpoint(self,resume_path):
        pass
    
    def save_checkpoint(self):
        pass

    def _valid_epoch(self):
        pass
    
