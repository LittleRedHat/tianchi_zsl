import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
from torch.backends import cudnn
import numpy as np
import argparse
import sys
sys.path.append('..')
from data.ZSLDataset import ZSLDataset
from logger.logger import SummaryLogger
from trainer.feat_trainer import FeatTrainer


def args_parser():
    parser = argparse.ArgumentParser("Tianchi ZSL Parameters")
    parser.add_argument('--epoches', type=int, help="epoch", default=100)
    parser.add_argument('--device', type=str, help="device", default='cuda')
    parser.add_argument('--device_ids',type=str,help='gpu device ids',default='0,1,2,3')
    parser.add_argument('--checkpoint',type=str,help='model dir')

    parser.add_argument('--log_frq', type=int, help="log iter", default=10)
    parser.add_argument('--eval_frq', type=int, help="model eval frequency", default=1)

    parser.add_argument('--train_file',type=str,help='train datas')
    parser.add_argument('--val_file',type=str,help='val datas')
    parser.add_argument('--lable_name_file',type=str,help='label name file')
    parser.add_argument('--image_dir',type=str,help='image dir')
    parser.add_argument('--embed_file',type=str,help='embed file')
    parser.add_argument('--attribute_file',type=str,help="attribute file")


    parser.add_argument('--save_dir', type=str, help="model save dir", default='zsl')

    parser.add_argument('--nw',type=int,help="number of workers",default=8)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=64)


    
    parser.add_argument('--seed', type=int, help="seed", default=-1)
    parser.add_argument('--backbone',type=str,default='mobilenet')

    parser.add_argument('--optim',type=str,default='adam',help="optimizer type")
    parser.add_argument('--optim_alpha', type=float, help="alpha", default=0.9)
    parser.add_argument('--optim_beta', type=float, help="beta", default=0.999)
    parser.add_argument('--optim_epsilon', type=float, help="epsilon", default=1e-8)
    parser.add_argument('--lr', type=float, help="learning_rate", default=1e-3)

    parser.add_argument('--patience', type=float, help="patience", default=2)
    parser.add_argument('--factor', type=float, help="learning rate decay", default=0.8)
    parser.add_argument('--weight_decay',type=float,default=0.0005)

    args = parser.parse_args()
    args.save_dir = os.path.join('./output/models',args.save_dir)
    setattr(args,'log_dir',os.path.join(args.save_dir,'log'))
    if args.seed == -1:
        args.seed = random.randint(-2^30,2^30)
    return args

image_size = (224,224)

def main():
    args = args_parser()
    utils.delete_path(args.log_dir)
    utils.ensure_path(args.save_dir)
    utils.ensure_path(args.log_dir)

    utils.write_dict(vars(args), os.path.join(args.save_dir, 'arguments.csv'))

    torch.manual_seed(args.seed)
    cudnn.benchmark = True 
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    

    train_transformer = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.RandomCrop(size=image_size),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = ZSLDataset(args.image_dir,args.train_file,args.embed_file,args.attribute_file,args.lable_name_file)
    train_dataloader = train_dataset.get_dataloader(batch_size=args.batch_size,num_workers=args.nw,shuffle=True)
    
    val_transformer = transforms.Compose([
        transforms.Resize(size=image_size),
        # transforms.CenterCrop(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_dataset = ZSLDataset(args.image_dir,args.val_file,args.embed_file,args.attribute_file,args.lable_name_file)
    val_dataloader =val_dataloader.get_dataloader(batch_size=args.batch_size,num_workers=args.nw,shuffle=False)

    print(vars(args))

    num_classes = train_dataset.get_class_size()


    train_logger = SummaryLogger(args.log_dir)

    trainer = FeatTrainer(num_classes,backbone=args.backbone,input_size=image_size,train_logger=train_logger,opt=args)
    trainer.train(train_dataloader,val_dataloader)





if __name__ == '__main__':
    main()