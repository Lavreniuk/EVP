import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models_depth.model import EVPDepth
from configs.train_options import TrainOptions
from configs.test_options import TestOptions
import glob
import utils
import torchvision.transforms as transforms
from utils_depth.misc import colorize
from PIL import Image
import torch.nn.functional as F


def main():
    opt = TestOptions().initialize()
    opt.add_argument('--img_path',  type=str)
    args = opt.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EVPDepth(args=args, caption_aggregation=True)
    cudnn.benchmark = True
    model.to(device)
    model_weight = torch.load(args.ckpt_dir)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
    model.eval()
    
    img_path = args.img_path
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0).to(device)
    shape = image.shape
    image = torch.nn.functional.interpolate(image, (440,480), mode='bilinear', align_corners=True)
    image = F.pad(image, (0, 0, 40, 0))

    with torch.no_grad():
        pred = model(image)['pred_d']
    
    pred = pred[:,:,40:,:] 
    pred = torch.nn.functional.interpolate(pred, shape[2:], mode='bilinear', align_corners=True)
    pred_d_numpy = pred.squeeze().cpu().numpy()
    pred_d_color, _, _ = colorize(pred_d_numpy, cmap='gray_r')
    Image.fromarray(pred_d_color).save('res.png')

    return 0

if __name__ == '__main__':
    main()
