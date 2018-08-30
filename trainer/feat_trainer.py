import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import sys
sys.path.append('..')
from model.features import MobileNet

import model.loss as losses
import model.metric as metrics

class FeatTrainer:
    def __init__(self,num_classes,backbone='mobilenet',input_size=(224,224),train_logger=None,opt=None):
        
        self.num_classes = num_classes
        

        self.opt = opt

        self.logger = logging.getLogger(__class__)

        self.train_logger = train_logger

        self.model = self._build_model(backbone,num_classes)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.optimizer = self._build_optimizer(self.model.parameters(),opt)

        self.scheduler = self._get_scheduler(self.optimizer,opt)

        self.start_epoch = 1
        self.best_acc = 0.0
        self.best_epoch = 1

        self._resume_checkpoint(opt.checkpoint)

        self.loss = nn.CrossEntropyLoss(size_average=True)



    def _build_optimizer(params, opt):
        if opt.optim == 'rmsprop':
            return optim.RMSprop(params, opt.lr, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
        elif opt.optim == 'adagrad':
            return optim.Adagrad(params, opt.lr, weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            return optim.SGD(params, opt.lr, weight_decay=opt.weight_decay)
        elif opt.optim == 'sgdm':
            return optim.SGD(params, opt.lr, opt.optim_alpha, weight_decay=opt.weight_decay)
        elif opt.optim == 'sgdmom':
            return optim.SGD(params, opt.lr, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
        elif opt.optim == 'adam':
            return optim.Adam(params, opt.lr, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
        else:
            raise Exception("bad option opt.optim: {}".format(opt.optim))
        
    


    def _build_model(self,backbone,num_classes,input_size=(224,224)):
        if backbone == 'mobilenet':
            model = MobileNet(num_classes,input_size)
        return model

    
    def _get_scheduler(self,optimizer,opt):

        if opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.every_epoches_decay, gamma=opt.gamma)

        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.factor, threshold=0.01, patience=opt.patience)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler





    def train(self,train_dataloader,val_dataloader=None):

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(self.model).cuda()
        else:
            model = self.model

        best_acc = self.best_acc
        best_epoch = self.start_epoch

        val_file = open(os.path.join(self.opt.save_dir,'val_result.csv'),'w')



        for epoch in range(self.start_epoch,self.epoches + 1):
            result = self._train_epoch(model,train_dataloader,self.opt)
            self.scheduler(result['val_loss'])

            self.train_logger.add_summary_value('train_loss',result['train_loss'],epoch)
            self.train_logger.add_summary_value('val_loss',result['val_loss'],epoch)
            self.train_logger.add_summary_value('val_acc',result['val_acc'],epoch)
            self.train_logger.add_summary_value('lr',self.optimizer.param_groups[0]['lr'],epoch)

            val_file.write('acc {} and loss {}\n'.format(result['val_acc'],result['val_loss']))
            val_file.flush()

            if best_acc < result['val_acc']:
                best_acc = result['val_auc']
                best_epoch = epoch

                info_to_save = {
                    'model':self.model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'lr':optimizer.param_groups[0]['lr'],
                    'epoch':epoch,
                    'best_acc':best_auc
                }
                torch.save(info_to_save,os.path.join(self.config.save_dir,'model_best.pth.tar'))



    def _train_epoch(self,model,train_dataloader,opt={}):
        start_time = time.time()
        model.train()
        sum_loss = 0.0
        result = {}
        for step,sample in enumerate(train_dataloader,1):
            self.optimizer.zero_grad()
            batch_images,batch_labels,_,_ = sample

            if torch.cuda.is_available():
                batch_images = batch_images.cuda()
                batch_labels = batch_labels.cuda()

            logits,_ = model(batch_images)
            loss = self.loss(logits,batch_labels)

            _loss = loss.data.cpu().numpy() if torch.cuda.is_available() else loss.data.numpy()
            sum_loss += _loss
            loss.backward()
            self.optimizer.step()

            if step % opt.log_frq == 0 or step == len(train_dataloader):
                self.logger.info('Train Epoch: {} {}/{} Loss: {:.6f} {:.4f} mins'.format(
                    epoch,
                    step,
                    len(train_dataloader),
                    _loss,
                    (time.time() - start_time) / 60.0
                    )
                )

        
        result = {'train_loss':sum_loss / len(train_dataloader)}

        if epoch % opt.eval_frq == 0 or epoch == opt.epoches:
            loss,acc = self._eval_epoch(model,val_dataloader)

            log = 'Eval Epoch: {} Loss: {:.6f} acc {:.5f}'.format(epoch,loss,acc)
            self.logger.info(log)
            result['val_acc'] = acc
            result['val_loss'] = loss

        return result

    def _resume_checkpoint(self,resume_path):
        if resume_path is not None:
            resumed_model = torch.load(resume_path)
            self.model.load_state_dict(resumed_model['model'])
            self.optimizer.load_state_dict(resumed_model['optimizer'])
            self.best_acc = resumed_model['best_acc']
            self.start_epoch = resumed_model['epoch']
        


    def _valid_epoch(self,model,val_dataloader):

        model.eval()
        targets = []
        preds = []

        with torch.no_grad():
            for sample in val_dataloader:
                batch_images,batch_labels,_,_ = sample
                if torch.cuda.is_available():
                    batch_images = batch_images.cuda()
                    batch_labels = batch_labels.cuda()

                logits,_ = model(batch_images)

                pred = F.softmax(logits,dim=1)


                if len(targets):
                    preds = np.concatenate((preds,pred.data.cpu().numpy()),0)
                    targets = np.concatenate((targets,batch_labels.data.cpu().numpy()),0)
                else:
                    targets = batch_labels.data.cpu().numpy()
                    preds = pred.data.cpu().numpy()

        acc = metrics.compute_acc(preds,targets)
        loss = losses.compute_cross_loss(preds,targets)
       
        return loss,acc
        
    
