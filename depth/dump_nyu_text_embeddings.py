import argparse
import copy
import os
import os.path as osp
import time
import torch
# import denseclip
from  tqdm import tqdm
from glob import glob
import json
from lavis.models import load_model_and_preprocess
from PIL import Image
import numpy as np

    
dataset = 'nyu' # 'kitti', 'nyu'


def main():
    import sys
    sys.path.append('../')
    from evp.models import FrozenCLIPEmbedder
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
	
    model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
            )

    if dataset == 'nyu':
        paths = glob('nyu_depth_v2/official_splits/train/*')
        paths_jpg = []
        file_paths = [
                    'dataset/filenames/nyudepthv2/train_list.txt',
                    'dataset/filenames/nyudepthv2/test_list.txt'
                     ]
        
        class_name = [path.split('/')[-1] for path in paths]
        #with open('nyu_class_list.json', 'w') as f:
        #    f.write(json.dumps(class_name))
        imagenet_classes = [name.replace('_', ' ') for name in class_name]
        classnames = imagenet_classes
        
        for i, file_path in enumerate(file_paths):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # Split the line and extract the first path
                    first_path = line.split()[0]
                    if i==0:
                        paths_jpg.append(os.path.join('nyu_depth_v2/sync', first_path.lstrip('/')))
                    else:
                        paths_jpg.append(os.path.join('nyu_depth_v2/official_splits/test', first_path.lstrip('/')))

    elif dataset == 'kitti':
        paths_jpg = []
        file_paths = [
                    'dataset/filenames/eigen_benchmark/train_list.txt',
                    'dataset/filenames/eigen_benchmark/test_list.txt'
                     ]

        for file_path in file_paths:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # Split the line and extract the first path
                    first_path = line.split()[0]
                    paths_jpg.append(os.path.join('kitti_dataset', first_path.lstrip('/')))
					
	
    # imagenet_classes = ['close',  'far', 'nearby', 'in middle distance', 'far away', 'in background']
    # mid = ['object ', 'something ', 'stuff ', 'guy ', 'people ']
	
    imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
    ]
	
    text_encoder = FrozenCLIPEmbedder(max_length=20)
    text_encoder.cuda()

    if dataset == 'nyu':
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = []
                texts = texts + [template.format(classname) for template in imagenet_templates] #format with class
                class_embeddings = text_encoder.encode(texts).detach().mean(dim=0)
                zeroshot_weights.append(class_embeddings)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
            mean_weights = torch.mean(zeroshot_weights, dim=0)
            zeroshot_weights = torch.cat((zeroshot_weights, mean_weights.unsqueeze(0)), dim=0)

        print(zeroshot_weights.shape)
        torch.save(zeroshot_weights.cpu(), 'nyu_class_embeddings_with_mean_weights.pth')

    zeroshot_weights = {}

    with torch.no_grad():
        for name in tqdm(paths_jpg):
            raw_image = Image.open(name)
            if dataset == 'kitti':
                image = np.array(raw_image)
                height, width = image.shape[0], image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                raw_image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                
                image = vis_processors["eval"](Image.fromarray(raw_image)).unsqueeze(0).to(device)
                caption = model.generate({"image": image})
                print(name, caption)
                class_embeddings = text_encoder.encode(caption).detach().mean(dim=0)
                zeroshot_weights[name] = {'class_embeddings' : class_embeddings, 'caption' : caption}

            else:
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                caption = model.generate({"image": image})
                print(name, caption)
                class_embeddings = text_encoder.encode(caption).detach().mean(dim=0)           
                zeroshot_weights[name] = {'class_embeddings' : class_embeddings, 'caption' : caption}

    torch.save(zeroshot_weights, f'{dataset}_class_embeddings_my_captions.pth')

if __name__ == '__main__':
    main()
