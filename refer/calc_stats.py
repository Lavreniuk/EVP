import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import warnings
warnings.filterwarnings("ignore")

from models_refer.model import EVPRefer

import transforms as T
import utils
import numpy as np

import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPTextModel
import gc
from collections import OrderedDict
import wandb
from timm.scheduler import CosineLRScheduler


def get_dataset(image_set, transform, args):
    from data.dataset_refer_clip import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=(image_set == 'val')
                      )
    num_classes = 2

    return ds, num_classes


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                  ]

    return T.Compose(transforms)


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, clip_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        
        embedding = clip_model(input_ids=sentences).last_hidden_state
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
        output = model(image, embedding)

        loss = criterion(output, target)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        
        lr_scheduler.step(epoch + total_its/len(data_loader))
        
        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        del embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        
def calc_mean(train_loader, model, device, args):    
    model.eval()
    
    sum_ = 0
    num_ = 0
    
    for batch_idx, batch in enumerate(train_loader):      
        
        img, target, sentences, attentions = batch
        input_RGB = img.to(device)
        with torch.no_grad():
            sum_ += torch.mean(model.get_latent(input_RGB))
            num_ += 1
        
        if utils.is_main_process():
            if batch_idx % args.print_freq == 0:
                print(batch_idx, sum_/num_)
            
    return sum_/num_


def calc_std(train_loader, model, device, args, mean):    
    model.eval()
    
    sum_ = 0
    num_ = 0
    
    for batch_idx, batch in enumerate(train_loader):      
        
        img, target, sentences, attentions = batch
        input_RGB = img.to(device)
        with torch.no_grad():
            sum_ += torch.mean(torch.square(model.get_latent(input_RGB) - mean))
            num_ += 1
        
        if utils.is_main_process():
            if batch_idx % args.print_freq == 0:
                print(batch_idx, torch.sqrt(sum_/num_), 1/torch.sqrt(sum_/num_))
            
    return torch.sqrt(sum_/num_)
    
    
def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    
    # data loader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    model = EVPRefer(sd_path='../checkpoints/v1-5-pruned-emaonly.ckpt',
                      neck_dim=[320,640+args.token_length,1280+args.token_length,1280]
                      )

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    device = torch.device("cuda:0")
    
    mean = calc_mean(train_loader, model, device=device, args=args)                       
    std = calc_std(train_loader, model, device=device, args=args, mean=mean)
                       
    print(f'mean = {mean}, std = {std}')
    

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
