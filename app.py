import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'depth')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'refer')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'stable-diffusion')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'taming-transformers')))

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'depth')))

import cv2
import numpy as np
import torch
from depth.models_depth.model import EVPDepth
from models_refer.model import EVPRefer
from depth.configs.train_options import TrainOptions
from depth.configs.test_options import TestOptions
import glob
import utils
import torchvision.transforms as transforms
from utils_depth.misc import colorize
from PIL import Image
import torch.nn.functional as F
import gradio as gr
import tempfile
from transformers import CLIPTokenizer


css = """
#img-display-container {
    max-height: 50vh;
    }
#img-display-input {
    max-height: 40vh;
    }
#img-display-output {
    max-height: 40vh;
    }

"""

def create_depth_demo(model, device):
    gr.Markdown("### Depth Prediction demo")
    with gr.Row():
        input_image = gr.Image(label="Input Image", type='pil', elem_id='img-display-input')
        depth_image = gr.Image(label="Depth Map", elem_id='img-display-output')
    raw_file = gr.File(label="16-bit raw depth, multiplier:256")
    submit = gr.Button("Submit")
    
    def on_submit(image):
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
        colored_depth, _, _ = colorize(pred_d_numpy, cmap='gray_r')
        
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth = Image.fromarray((pred_d_numpy*256).astype('uint16'))
        raw_depth.save(tmp.name)
        return [colored_depth, tmp.name]

    submit.click(on_submit, inputs=[input_image], outputs=[depth_image, raw_file])
    examples = gr.Examples(examples=["imgs/test_img1.jpg", "imgs/test_img2.jpg", "imgs/test_img3.jpg", "imgs/test_img4.jpg", "imgs/test_img5.jpg"],
                           inputs=[input_image])


def create_refseg_demo(model, tokenizer, device):
    gr.Markdown("### Referring Segmentation demo")
    with gr.Row():
        input_image = gr.Image(label="Input Image", type='pil', elem_id='img-display-input')
        refseg_image = gr.Image(label="Output Mask", elem_id='img-display-output')
    input_text = gr.Textbox(label='Prompt', placeholder='Please upload your image first', lines=2)
    submit = gr.Button("Submit")
    
    def on_submit(image, text):
        image = np.array(image)
        image_t = transforms.ToTensor()(image).unsqueeze(0).to(device)
        image_t = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_t)
        shape = image_t.shape
        image_t = torch.nn.functional.interpolate(image_t, (512,512), mode='bilinear', align_corners=True)
        input_ids = tokenizer(text=text, truncation=True, max_length=40, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")['input_ids'].to(device)

        with torch.no_grad():
            pred = model(image_t, input_ids)

        pred = torch.nn.functional.interpolate(pred, shape[2:], mode='bilinear', align_corners=True)
        output_mask = pred.cpu().argmax(1).data.numpy().squeeze()
        alpha = 0.65
        image[output_mask == 0] = (image[output_mask == 0]*alpha).astype(np.uint8)
        contours, _ = cv2.findContours(output_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        return Image.fromarray(image)

    submit.click(on_submit, inputs=[input_image, input_text], outputs=refseg_image)
    examples = gr.Examples(examples=[["imgs/test_img2.jpg", "green plant"], ["imgs/test_img3.jpg", "chair"], ["imgs/test_img4.jpg", "left green plant"], ["imgs/test_img5.jpg", "man walking on foot"], ["imgs/test_img5.jpg", "the rightest camel"]],
                           inputs=[input_image, input_text])
                           

def main():
    upload_2_models = True

    opt = TestOptions().initialize()
    args = opt.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if upload_2_models:
        model = EVPDepth(args=args, caption_aggregation=True)
        model.to(device)
        model_weight = torch.load('best_model_nyu.ckpt', map_location=device)['model']
        model.load_state_dict(model_weight, strict=False)
        model.eval()
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    model_refseg = EVPRefer()
    model_refseg.to(device)
    model_weight = torch.load('best_model_refcoco.pth', map_location=device)['model']
    model_refseg.load_state_dict(model_weight, strict=False)
    model_refseg.eval()
    
    del model_weight
    print('Models uploaded successfully')
    
    title = "# EVP"
    description = """Official demo for **EVP: Enhanced Visual Perception using Inverse Multi-Attentive Feature
    Refinement and Regularized Image-Text Alignment**.
    EVP is a deep learning model for metric depth estimation from a single image as well as referring segmentation.
    Please refer to our [project page](https://lavreniuk.github.io/EVP) or [paper](https://arxiv.org/abs/2312.08548) or [github](https://github.com/Lavreniuk/EVP) for more details."""

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        if upload_2_models:
            with gr.Tab("Depth Prediction"):
                create_depth_demo(model, device)
        with gr.Tab("Referring Segmentation"):
            create_refseg_demo(model_refseg, tokenizer, device)
        gr.HTML('''<br><br><br><center>You can duplicate this Space to skip the queue:<a href="https://huggingface.co/spaces/MykolaL/evp?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a><br>
                <p><img src="https://visitor-badge.glitch.me/badge?page_id=MykolaL/evp" alt="visitors"></p></center>''')

    demo.queue().launch(share=True)


if __name__ == '__main__':
    main()
