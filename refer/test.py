import datetime
import os
import time
import cv2
import torch
import torch.utils.data
from torch import nn
from torchvision import transforms
import transforms as T
import utils
from transformers.models.clip.modeling_clip import CLIPTextModel
from models_refer.model import EVPRefer
import numpy as np
from PIL import Image
import torch.nn.functional as F


def get_dataset(image_set, transform, args):
    from data.dataset_refer_clip import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        for i, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()

            for idx in range(sentences.size(-1)):
                
                #embedding = clip_model(input_ids=sentences[:, :, idx]).last_hidden_state
                attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
                output = model(image, sentences[:, :, idx])
                output = output.cpu()

                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1
                
            # save visualize
            if not os.path.exists('results'):
                os.makedirs('results')
            save_path = f'results/{i}.png'
                        
            output_mask = output_mask.squeeze()
            target = target.squeeze()
            h, w = output_mask.shape
            
            denormalize = transforms.Compose([
                    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
                    transforms.ToPILImage()
                        ])
            
            rgb_image = np.array(denormalize(image.squeeze()))
            
            rgb_image_gt = rgb_image.copy()
            alpha = 0.65
            rgb_image[output_mask == 0] = (rgb_image[output_mask == 0]*alpha).astype(np.uint8)
            rgb_image_gt[target == 0] = (rgb_image_gt[target == 0]*alpha).astype(np.uint8)
            contours, _ = cv2.findContours(output_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_image, contours, -1, (0, 255, 0), 2)
            contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_image_gt, contours, -1, (0, 255, 0), 2)

            Image.fromarray(rgb_image.astype(np.uint8)).save(save_path)
            Image.fromarray(rgb_image_gt.astype(np.uint8)).save(save_path.replace('.png','_gt.png'))

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    
    single_model = EVPRefer(sd_path='../checkpoints/v1-5-pruned-emaonly.ckpt',
                      neck_dim=[320,640+args.token_length,1280+args.token_length,1280]
                      )

    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)
    evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
