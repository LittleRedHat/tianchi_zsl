import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
sys.path.append('..')
from model.features import MobileNetMLC

import model.loss as losses

class FeatTrainer:
    def __init__(self,num_classes,backbone='mobilenet',train_logger=None,checkpoint=None,opt=None):
        
        self.num_classes = num_classes
        

        self.opt = opt

        self.logger = logging.getLogger(__class__)

        self.train_logger = train_logger

        self.model = self._build_model(backbone,num_classes)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.optimizer = self._build_optimizer(self.model.parameters(),opt)

        self.scheduler = self._get_scheduler(self.optimizer,opt)

        self._resume_checkpoint(checkpoint)

        self.loss = nn.CrossEntropyLoss(size_average=True)



    def _build_optimizer(params, opt):
        if opt.optim == 'rmsprop':
            return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
        elif opt.optim == 'adagrad':
            return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
        elif opt.optim == 'sgdm':
            return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
        elif opt.optim == 'sgdmom':
            return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
        elif opt.optim == 'adam':
            return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
        else:
            raise Exception("bad option opt.optim: {}".format(opt.optim))
        
    


    def _build_model(self,backbone,num_classes):
        if backbone == 'mobilenet':
            model = MobileNetMLC(num_classes,input_size)
        return model

    
    def _get_scheduler(self,optimizer,opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler





    def train(self,train_dataloader,val_dataloader=None):

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(self.model).cuda()
        else:
            model = self.model

        best_acc = 0.0
        best_epoch = self.best_epoch

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

            if best_auc < result['val_mean_auc']:
                best_auc = result['val_mean_auc']
                best_epoch = epoch

                info_to_save = {
                    'model':self.model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'lr':optimizer.param_groups[0]['lr'],
                    'epoch':epoch,
                    'best_auc':best_auc
                }
                self.save_checkpoint()
                torch.save(info_to_save,os.path.join(self.config.save_dir,'model_best.pkl'))



    def _train_epoch(self,model,train_dataloader,opt={}):
        start_time = time.time()
        model.train()
        sum_loss = 0.0
        result = {}
        for step,sample in enumerate(train_dataloader,1):
            self.optimizer.zero_grad()
            batch_images,batch_labels,_,_ = sample
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
            loss,auc,_,_ = self._eval_epoch(model,val_dataloader)
            _auc = [str(i) for i in auc]
            _auc = ','.join(_auc)
            mean_auc = np.mean(auc)
            log = 'Eval Epoch: {} Loss: {:.6f} auc {} mean_auc {:.5f}'.format(epoch,loss,_auc,mean_auc)
            self.logger.info(log)
            result['val_auc'] = auc
            result['val_mean_auc'] = mean_auc
            result['val_loss'] = loss

        




        
    

    def _resume_checkpoint(self,resume_path):
        if resume_path is not None:
            pass
        
    
    def save_checkpoint(self):
        pass

    def _valid_epoch(self):
        pass
    
