import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models_depth.model import EVPDepth
from models_depth.optimizer import build_optimizers
import utils_depth.metrics as metrics
from utils_depth.criterion import SiLogLoss
import utils_depth.logging as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
import glob
import utils
import wandb
from timm.scheduler import CosineLRScheduler


def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()

    utils.init_distributed_mode_simple(args)
    device = torch.device(args.gpu)

    pretrain = args.pretrained.split('.')[0]
    maxlrstr = str(args.max_lr).replace('.', '')
    minlrstr = str(args.min_lr).replace('.', '')
    layer_decaystr = str(args.layer_decay).replace('.', '')
    weight_decaystr = str(args.weight_decay).replace('.', '')
    num_filter = str(args.num_filters[0]) if args.num_deconv > 0 else ''
    num_kernel = str(args.deconv_kernels[0]) if args.num_deconv > 0 else ''
    name = [args.dataset, str(args.batch_size), pretrain.split('/')[-1], 'deconv'+str(args.num_deconv), \
        str(num_filter), str(num_kernel), str(args.crop_h), str(args.crop_w), maxlrstr, minlrstr, \
        layer_decaystr, weight_decaystr, str(args.epochs)]
    if args.exp_name != '':
        name.append(args.exp_name)

    exp_name = '_'.join(name)
    print('This experiments: ', exp_name)
    
    model = EVPDepth(args=args).encoder

    # CPU-GPU agnostic settings
    
    cudnn.benchmark = True
    model.to(device)

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    train_dataset = get_dataset(**dataset_kwargs)

    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=True, 
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=sampler_train, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True)
    
    # Training settings
    criterion_d = SiLogLoss()

    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                constructor='LDMOptimizerConstructor',
                paramwise_cfg=dict(layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

    log_txt = epoch = ''
    # Perform experiment
    mean = calc_mean(train_loader, model, criterion_d, log_txt, optimizer=optimizer, 
                       device=device, epoch=epoch, args=args)
                       
    std = calc_std(train_loader, model, criterion_d, log_txt, optimizer=optimizer, 
                       device=device, epoch=epoch, args=args, mean=mean)
                       
    print(f'mean = {mean}, std = {std}')
    

def calc_mean(train_loader, model, criterion_d, log_txt, optimizer, device, epoch, args):    
    model.eval()
    
    sum_ = 0
    num_ = 0
    
    for batch_idx, batch in enumerate(train_loader):      
        
        input_RGB = batch['image'].to(device)
        with torch.no_grad():
            sum_ += torch.mean(model.get_latent(input_RGB))
            num_ += 1
        
        if args.rank == 0:
            if batch_idx % args.print_freq == 0:
                print(batch_idx, sum_/num_)
            
    return sum_/num_


def calc_std(train_loader, model, criterion_d, log_txt, optimizer, device, epoch, args, mean):    
    model.eval()
    
    sum_ = 0
    num_ = 0
    
    for batch_idx, batch in enumerate(train_loader):      
        
        input_RGB = batch['image'].to(device)
        with torch.no_grad():
            sum_ += torch.mean(torch.square(model.get_latent(input_RGB) - mean))
            num_ += 1
        
        if args.rank == 0:
            if batch_idx % args.print_freq == 0:
                print(batch_idx, torch.sqrt(sum_/num_), 1/torch.sqrt(sum_/num_))
            
    return torch.sqrt(sum_/num_)

if __name__ == '__main__':
    main()
