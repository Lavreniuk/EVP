import os
import cv2
import numpy as np
import torch
from models_refer.model import EVPRefer
from args import get_parser
import glob
import utils
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPTokenizer


def main():
    parser = get_parser()
    parser.add_argument('--img_path',  type=str)
    parser.add_argument('--prompt',  type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    model = EVPRefer(sd_path='../checkpoints/v1-5-pruned-emaonly.ckpt')
    model.to(device)
    model_weight = torch.load(args.resume)['model']
    model.load_state_dict(model_weight, strict=False)
    model.eval()
    
    img_path = args.img_path
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_t = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image_t = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_t)
    shape = image_t.shape
    image_t = torch.nn.functional.interpolate(image_t, (512,512), mode='bilinear', align_corners=True)
    input_ids = tokenizer(text=args.prompt, truncation=True, max_length=args.token_length, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")['input_ids'].to(device)
    
    with torch.no_grad():
        pred = model(image_t, input_ids)
    
    pred = torch.nn.functional.interpolate(pred, shape[2:], mode='bilinear', align_corners=True)
    output_mask = pred.cpu().argmax(1).data.numpy().squeeze()
    
    alpha = 0.65
    image[output_mask == 0] = (image[output_mask == 0]*alpha).astype(np.uint8)
    contours, _ = cv2.findContours(output_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    Image.fromarray(image.astype(np.uint8)).save('res.png')

    return 0

if __name__ == '__main__':
    main()
