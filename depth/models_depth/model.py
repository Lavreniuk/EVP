# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The deconvolution code is based on Simple Baseline.
# (https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch.nn.functional as F

from evp.models import UNetWrapper, TextAdapterRefer, FrozenCLIPEmbedder
from .miniViT import mViT
from .attractor import AttractorLayer, AttractorLayerUnnormed
from .dist_layers import ConditionalLogBinomial
from .localbins_layers import (Projector, SeedBinRegressor, SeedBinRegressorUnnormed)
import os


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)
    
    
class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x
        

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Group Normalization
        self.group_norm = nn.GroupNorm(20, out_channels)
        
        # ReLU Activation
        self.relu = nn.ReLU()
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply spatial attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        
        # Apply convolutional layer
        x = self.conv1(x)
        x = self.group_norm(x)
        x = self.relu(x)
        
        return x
        
        
class AttentionDownsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(AttentionDownsamplingModule, self).__init__()
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Convolutional Layers
        if scale_factor == 2:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        elif scale_factor == 4:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        # Group Normalization
        self.group_norm = nn.GroupNorm(20, out_channels)
        
        # ReLU Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Apply spatial attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        
        # Apply channel attention
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.group_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.group_norm(x)
        x = self.relu(x)
        
        return x


class AttentionUpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpsamplingModule, self).__init__()
        
        # Spatial Attention for outs[2]
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel Attention for outs[2]
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Group Normalization
        self.group_norm = nn.GroupNorm(20, out_channels)
        
        # ReLU Activation
        self.relu = nn.ReLU()
        self.upscale = PixelShuffle(in_channels, 2)
        
    def forward(self, x):
        # Apply spatial attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        
        # Apply channel attention
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.group_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.group_norm(x)
        x = self.relu(x)
        
        # Upsample
        x = self.upscale(x)
        
        return x
        
        
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(20, out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        
        return x
        
        
class InverseMultiAttentiveFeatureRefinement(nn.Module):
    def __init__(self, in_channels_list):
        super(InverseMultiAttentiveFeatureRefinement, self).__init__()
        
        self.layer1 = AttentionModule(in_channels_list[0], in_channels_list[0])
        self.layer2 = AttentionDownsamplingModule(in_channels_list[0], in_channels_list[0]//2, scale_factor = 2)
        self.layer3 = ConvLayer(in_channels_list[0]//2 + in_channels_list[1], in_channels_list[1])
        self.layer4 = AttentionDownsamplingModule(in_channels_list[1], in_channels_list[1]//2, scale_factor = 2)
        self.layer5 = ConvLayer(in_channels_list[1]//2 + in_channels_list[2], in_channels_list[2])
        self.layer6 = AttentionDownsamplingModule(in_channels_list[2], in_channels_list[2]//2, scale_factor = 2)
        self.layer7 = ConvLayer(in_channels_list[2]//2 + in_channels_list[3], in_channels_list[3])
        
        '''
        self.layer8 = AttentionUpsamplingModule(in_channels_list[3], in_channels_list[3])
        self.layer9 = ConvLayer(in_channels_list[2] + in_channels_list[3], in_channels_list[2])
        self.layer10 = AttentionUpsamplingModule(in_channels_list[2], in_channels_list[2])
        self.layer11 = ConvLayer(in_channels_list[1] + in_channels_list[2], in_channels_list[1])
        self.layer12 = AttentionUpsamplingModule(in_channels_list[1], in_channels_list[1])
        self.layer13 = ConvLayer(in_channels_list[0] + in_channels_list[1], in_channels_list[0])
        '''
    def forward(self, inputs):
        x_c4, x_c3, x_c2, x_c1 = inputs
        x_c4 = self.layer1(x_c4)
        x_c4_3 = self.layer2(x_c4)
        x_c3 = torch.cat([x_c4_3, x_c3], dim=1)
        x_c3 = self.layer3(x_c3)
        x_c3_2 = self.layer4(x_c3)
        x_c2 = torch.cat([x_c3_2, x_c2], dim=1)
        x_c2 = self.layer5(x_c2)
        x_c2_1 = self.layer6(x_c2)
        x_c1 = torch.cat([x_c2_1, x_c1], dim=1)
        x_c1 = self.layer7(x_c1)
        '''
        x_c1_2 = self.layer8(x_c1)
        x_c2 = torch.cat([x_c1_2, x_c2], dim=1)
        x_c2 = self.layer9(x_c2)
        x_c2_3 = self.layer10(x_c2)
        x_c3 = torch.cat([x_c2_3, x_c3], dim=1)
        x_c3 = self.layer11(x_c3)
        x_c3_4 = self.layer12(x_c3)
        x_c4 = torch.cat([x_c3_4, x_c4], dim=1)
        x_c4 = self.layer13(x_c4)
        '''
        return [x_c4, x_c3, x_c2, x_c1]
        

class EVPDepthEncoder(nn.Module):
    def __init__(self, out_dim=1024, ldm_prior=[320, 680, 1320+1280], sd_path=None, text_dim=768,
                 dataset='nyu', caption_aggregation=False
                ):
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )
        
        self.aggregation = InverseMultiAttentiveFeatureRefinement([320, 680, 1320, 1280])

        self.apply(self._init_weights)

        ### stable diffusion layers 

        config = OmegaConf.load('./v1-inference.yaml')
        if sd_path is None:
            if os.path.exists('../checkpoints/v1-5-pruned-emaonly.ckpt'):
                config.model.params.ckpt_path = '../checkpoints/v1-5-pruned-emaonly.ckpt'
            else:
                config.model.params.ckpt_path = None
        else:
            config.model.params.ckpt_path = f'../{sd_path}'

        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model

        self.unet = UNetWrapper(sd_model.model, use_attn=True)
        if dataset == 'kitti':
            self.unet = UNetWrapper(sd_model.model, use_attn=True, base_size=384)
    
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.out
        del self.encoder_vq.post_quant_conv.weight
        del self.encoder_vq.post_quant_conv.bias

        for param in self.encoder_vq.parameters():
            param.requires_grad = True
        
        self.text_adapter = TextAdapterRefer(text_dim=text_dim)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if caption_aggregation:                
            class_embeddings = torch.load(f'{dataset}_class_embeddings_my_captions.pth', map_location=device)
            #class_embeddings_list = [value['class_embeddings'] for key, value in class_embeddings.items()]
            #stacked_embeddings = torch.stack(class_embeddings_list, dim=0)
            #class_embeddings = torch.mean(stacked_embeddings, dim=0).unsqueeze(0)
            
            if 'aggregated' in class_embeddings:
                class_embeddings = class_embeddings['aggregated'] 
            else:
                clip_model = FrozenCLIPEmbedder(max_length=40,pool=False).to(device)
                class_embeddings_new = [clip_model.encode(value['caption'][0]) for key, value in class_embeddings.items()]
                class_embeddings_new = torch.mean(torch.stack(class_embeddings_new, dim=0), dim=0)
                class_embeddings['aggregated'] = class_embeddings_new
                torch.save(class_embeddings, f'{dataset}_class_embeddings_my_captions.pth')
                class_embeddings = class_embeddings['aggregated']
            self.register_buffer('class_embeddings', class_embeddings)
        else:
            self.class_embeddings = torch.load(f'{dataset}_class_embeddings_my_captions.pth', map_location=device)

            self.clip_model = FrozenCLIPEmbedder(max_length=40,pool=False)
            for param in self.clip_model.parameters():
                param.requires_grad = True
                
        #if dataset == 'kitti':    
        #    self.text_adapter_ = TextAdapterRefer(text_dim=text_dim)
        #    self.gamma_ = nn.Parameter(torch.ones(text_dim) * 1e-4)
        
        self.caption_aggregation = caption_aggregation
        self.dataset = dataset
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x =  self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, x, class_ids=None, img_paths=None):
        latents = self.encoder_vq.encode(x).mode()
        
        # add division by std
        if self.dataset == 'nyu':
            latents = latents / 5.07543
        elif self.dataset == 'kitti':
            latents = latents / 4.6211
        else:
            print('Please calculate the STD for the dataset!')

        if class_ids is not None:
            if self.caption_aggregation:
                class_embeddings = self.class_embeddings[[0]*len(class_ids.tolist())]#[class_ids.tolist()]
            else:
                class_embeddings = []
                
                for img_path in img_paths:
                    class_embeddings.extend([value['caption'][0] for key, value in self.class_embeddings.items() if key in img_path.replace('//', '/')])
                    
                class_embeddings = self.clip_model.encode(class_embeddings)
        else:
            class_embeddings = self.class_embeddings
        
        c_crossattn = self.text_adapter(latents, class_embeddings, self.gamma)
        t = torch.ones((x.shape[0],), device=x.device).long()

        #if self.dataset == 'kitti':
        #    c_crossattn_last = self.text_adapter_(latents, class_embeddings, self.gamma_)
        #    outs = self.unet(latents, t, c_crossattn=[c_crossattn, c_crossattn_last])
        #else:
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        outs = self.aggregation(outs)
        
        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x)
    
    def get_latent(self, x):
        return self.encoder_vq.encode(x).mode()


class EVPDepth(nn.Module):
    def __init__(self, args=None, caption_aggregation=False):
        super().__init__()
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth_eval

        embed_dim = 192
        
        channels_in = embed_dim*8
        channels_out = embed_dim

        if args.dataset == 'nyudepthv2':
            self.encoder = EVPDepthEncoder(out_dim=channels_in, dataset='nyu', caption_aggregation=caption_aggregation)
        else:
            self.encoder = EVPDepthEncoder(out_dim=channels_in, dataset='kitti', caption_aggregation=caption_aggregation)
            
        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()
        self.mViT = False
        self.custom = False
        
        
        if not self.mViT and not self.custom:
            n_bins = 64
            bin_embedding_dim = 128
            num_out_features = [32, 32, 32, 192]
            min_temp = 0.0212
            max_temp = 50
            btlnck_features = 256
            n_attractors = [16, 8, 4, 1]
            attractor_alpha = 1000
            attractor_gamma = 2
            attractor_kind = "mean"
            attractor_type = "inv"
            self.bin_centers_type = "softplus"
                        
            self.bottle_neck = nn.Sequential(
                nn.Conv2d(channels_in, btlnck_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(btlnck_features, btlnck_features, kernel_size=3, stride=1, padding=1))
            
    
            for m in self.bottle_neck.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001, bias=0)
                        
            
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
            self.seed_bin_regressor = SeedBinRegressorLayer(
                btlnck_features, n_bins=n_bins, min_depth=self.min_depth, max_depth=self.max_depth)
            self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
            self.projectors = nn.ModuleList([
                Projector(num_out, bin_embedding_dim)
                for num_out in num_out_features
            ])
            self.attractors = nn.ModuleList([
                Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=self.min_depth, max_depth=self.max_depth,
                          alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
                for i in range(len(num_out_features))
            ])
            
            last_in = 192 + 1
            self.conditional_log_binomial = ConditionalLogBinomial(
                last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)
        elif self.mViT and not self.custom:
            n_bins = 256
            self.adaptive_bins_layer = mViT(192, n_query_channels=192, patch_size=16,
                                            dim_out=n_bins,
                                            embedding_dim=192, norm='linear')
            self.conv_out = nn.Sequential(nn.Conv2d(192, n_bins, kernel_size=1, stride=1, padding=0),
                                          nn.Softmax(dim=1))
                             
                                          
    def forward(self, x, class_ids=None, img_paths=None):
        b, c, h, w = x.shape
        x = x*2.0 - 1.0  # normalize to [-1, 1]
        if h == 480 and w == 480:
            new_x = torch.zeros(b, c, 512, 512, device=x.device)
            new_x[:, :, 0:480, 0:480] = x
            x = new_x
        elif h==352 and w==352:
            new_x = torch.zeros(b, c, 384, 384, device=x.device)
            new_x[:, :, 0:352, 0:352] = x
            x = new_x
        elif h == 512 and w == 512:
            pass
        else:
            print(h,w)
            raise NotImplementedError
        conv_feats = self.encoder(x, class_ids, img_paths)

        if h == 480 or h == 352:
            conv_feats = conv_feats[:, :, :-1, :-1]      
        
        self.decoder.remove_hooks()
        out_depth, out, x_blocks = self.decoder([conv_feats])
        
        if not self.mViT and not self.custom:
            x = self.bottle_neck(conv_feats)
            _, seed_b_centers = self.seed_bin_regressor(x)
            
            if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
                b_prev = (seed_b_centers - self.min_depth) / \
                    (self.max_depth - self.min_depth)
            else:
                b_prev = seed_b_centers
                
            prev_b_embedding = self.seed_projector(x)
            
            for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
                b_embedding = projector(x)
                b, b_centers = attractor(
                    b_embedding, b_prev, prev_b_embedding, interpolate=True)
                b_prev = b.clone()
                prev_b_embedding = b_embedding.clone()
            
            rel_cond = torch.sigmoid(out_depth) * self.max_depth
            
            # concat rel depth with last. First interpolate rel depth to last size
            rel_cond = nn.functional.interpolate(
                rel_cond, size=out.shape[2:], mode='bilinear', align_corners=True)
            last = torch.cat([out, rel_cond], dim=1)
                        
            b_embedding = nn.functional.interpolate(
                b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
            x = self.conditional_log_binomial(last, b_embedding)
            
            # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
            b_centers = nn.functional.interpolate(
                b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
            out_depth = torch.sum(x * b_centers, dim=1, keepdim=True)
                        
        elif self.mViT and not self.custom:
            bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(out)
            out = self.conv_out(range_attention_maps)
            
            bin_widths = (self.max_depth - self.min_depth) * bin_widths_normed  # .shape = N, dim_out
            bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
            bin_edges = torch.cumsum(bin_widths, dim=1)

            centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
            n, dout = centers.size()
            centers = centers.view(n, dout, 1, 1)

            out_depth = torch.sum(out * centers, dim=1, keepdim=True)
        else:
            out_depth = torch.sigmoid(out_depth) * self.max_depth
        
        return {'pred_d': out_depth}


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels
        
        embed_dim = 192
        
        channels_in = embed_dim*8
        channels_out = embed_dim
        
        self.deconv_layers, self.intermediate_results = self._make_deconv_layer(
            args.num_deconv,
            args.num_filters,
            args.deconv_kernels,
        )
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
            
        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)
        out = self.up(out)
        self.intermediate_results.append(out)
        out = self.up(out)
        out_depth = self.last_layer_depth(out)

        return out_depth, out, self.intermediate_results

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        intermediate_results = []  # List to store intermediate feature maps

        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU())
            in_planes = planes

            # Add a hook to store the intermediate result
            layers[-1].register_forward_hook(self._hook_fn(intermediate_results))

        return nn.Sequential(*layers), intermediate_results

    def _hook_fn(self, intermediate_results):
        def hook(module, input, output):
            intermediate_results.append(output)
        return hook
        
    def remove_hooks(self):
        self.intermediate_results.clear()
        
    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
