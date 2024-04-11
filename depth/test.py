# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from datetime import datetime
import tqdm
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
from collections import OrderedDict

from models_depth.model import EVPDepth
from utils_depth.misc import colorize
import utils_depth.metrics as metrics
import utils_depth.logging as logging

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
import utils

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main():
    opt = TestOptions()
    args = opt.initialize().parse_args()
    utils.init_distributed_mode_simple(args)
    device = torch.device(args.gpu)
        
    model = EVPDepth(args=args, caption_aggregation=False)

    # CPU-GPU agnostic settings
    
    cudnn.benchmark = True
    model.to(device)
    
    model_weight = torch.load(args.ckpt_dir)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
    model.eval()
    
    frozen_params = 0
    encoder_params = 0
    all_params = 0
    unet_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if not param.requires_grad:
            frozen_params += param.numel()
        if 'encoder_vq' in name:
            encoder_params += param.numel()
        if 'unet' in name:
            unet_params += param.numel()
    print(all_params, encoder_params, unet_params, frozen_params)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)


    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    val_dataset = get_dataset(**dataset_kwargs, is_train=False)


    sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=sampler_val,
                                             pin_memory=True)

    # Perform experiment

    results_dict = validate(val_loader, model,
                                        device=device, args=args)
    if args.rank == 0:
        result_lines = logging.display_result(results_dict)
        print(result_lines)



def validate(val_loader, model, device, args):
    
    if args.save_eval_pngs or args.save_visualize:
        result_path = os.path.join(args.result_dir, args.exp_name)
        if args.rank == 0:
            logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)

    if args.rank == 0:
        depth_loss = logging.AverageMeter()
    model.eval()

    ddp_logger = utils.MetricLogger()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in tqdm.tqdm(enumerate(val_loader)):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        filename = batch['filename'][0]
        class_id = batch['class_id']

        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                img_paths = []
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                    class_ids.append(class_id)
                    img_paths.extend(batch['img_path'])
                input_RGB = torch.cat(sliding_images, dim=0)
                class_ids = torch.cat(class_ids, dim=0)
            else:
                class_ids = class_id
                img_paths = batch['img_path']
                input_RGB = torch.nn.functional.interpolate(
                                    input_RGB, (480,480), mode='bilinear', align_corners=True)
            
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                class_ids = torch.cat((class_ids, class_ids), dim=0)
                img_paths.extend(img_paths)
            
            input_channels = input_RGB.split(1, dim=0)
            outputs_per_channel = []
            for input_channel in input_channels:

                pred = model(input_channel, class_ids=class_ids[0].unsqueeze(dim=0), img_paths=[img_paths[0]])
                outputs_per_channel.append(pred['pred_d'])
            pred = torch.cat(outputs_per_channel, dim=0)
        pred_d = pred

        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
            pred_d = pred_s/sliding_masks  
        else:
            pred_d = torch.nn.functional.interpolate(
                            pred_d, batch['image'].shape[2:], mode='bilinear', align_corners=True)

        pred_d = pred_d.squeeze()
        depth_gt = depth_gt.squeeze()

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
    
        if args.save_eval_pngs:
            save_path = os.path.join(result_path, filename)
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')
            pred_d = pred_d.squeeze()
            if args.dataset == 'nyudepthv2':
                pred_d = pred_d.cpu().numpy() * 1000.0
                cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                pred_d = pred_d.cpu().numpy() * 256.0
                cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        if args.save_visualize:
            save_path = os.path.join(result_path, filename)
            pred_d_numpy = pred_d.squeeze().cpu().numpy()        
            depth_gt_numpy = depth_gt.squeeze().cpu().numpy()
            batch_image = cv2.imread(batch['img_path'][0])
            batch_image = cv2.cvtColor(batch_image, cv2.COLOR_BGR2RGB)
            delta = np.abs(depth_gt_numpy - pred_d_numpy)
            
            depth_gt_color, vmin, vmax = colorize(depth_gt_numpy, cmap='magma_r', invalid_val=0)
            pred_d_color, _, _ = colorize(pred_d_numpy, vmin, vmax, cmap='magma_r', invalid_val=0)
            delta_color, _, _ = colorize(delta, vmin=0, vmax=1, cmap='Reds', invalid_mask=depth_gt_numpy==0)
            batch_image = np.concatenate([batch_image, 255 *np.ones((batch_image.shape[0], batch_image.shape[1], 1), dtype=np.uint8)], axis=2)
            
            if args.dataset == 'nyudepthv2':
                batch_image = batch_image[30:-10,10:-10,:]
                pred_d_color = pred_d_color[30:-10,10:-10,:]
                depth_gt_color = depth_gt_color[30:-10,10:-10,:]
                delta_color = delta_color[30:-10,10:-10,:]
            else:
                height, width = pred_d_color.shape[:2]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                batch_image = batch_image[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                pred_d_color = pred_d_color[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                depth_gt_color = depth_gt_color[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                delta_color = delta_color[top_margin:top_margin + 352, left_margin:left_margin + 1216]

            #pred_d = np.hstack((batch_image, depth_gt_color, pred_d_color, delta_color))
            
            pred_d = np.hstack((batch_image, depth_gt_color, pred_d_color))
            
            #pred_d = (pred_d_numpy*1000).astype(np.uint16)
            # save depth map only
            # Image.fromarray(pred_d_color).save(save_path.replace('.jpg','.png'))
            
            # save pair of depth map and rgb input image
            Image.fromarray(pred_d).save(save_path.replace('.jpg','.png'))
            
        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    # for key in result_metrics.keys():
    #     result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    return result_metrics


if __name__ == '__main__':
    main()
