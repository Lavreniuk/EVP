import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from ldm.util import instantiate_from_config
from transformers.models.clip.modeling_clip import CLIPTextModel
from omegaconf import OmegaConf
from lib.mask_predictor import SimpleDecoding

from evp.models import UNetWrapper, TextAdapterRefer


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
        
        
        
class EVPRefer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 sd_path=None,
                 base_size=512,
                 token_embed_dim=768,
                 neck_dim=[320,680,1320,1280],
                 **args):
        super().__init__()
        config = OmegaConf.load('./v1-inference.yaml')
        if os.path.exists(f'{sd_path}'):
            config.model.params.ckpt_path = f'{sd_path}'
        else:
            config.model.params.ckpt_path = None

        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model
        self.unet = UNetWrapper(sd_model.model, base_size=base_size)
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        for param in self.encoder_vq.parameters():
            param.requires_grad = True
            
        self.text_adapter = TextAdapterRefer(text_dim=token_embed_dim)

        self.classifier = SimpleDecoding(dims=neck_dim)

        self.gamma = nn.Parameter(torch.ones(token_embed_dim) * 1e-4)
        self.aggregation = InverseMultiAttentiveFeatureRefinement([320,680,1320,1280])
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        for param in self.clip_model.parameters():
            param.requires_grad = True


    def forward(self, img, sentences):
        input_shape = img.shape[-2:]
        
        latents = self.encoder_vq.encode(img).mode()
        latents = latents / 4.7164
        
        l_feats = self.clip_model(input_ids=sentences).last_hidden_state
        c_crossattn = self.text_adapter(latents, l_feats, self.gamma) # NOTE: here the c_crossattn should be expand_dim as latents
        t = torch.ones((img.shape[0],), device=img.device).long()
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        
        outs = self.aggregation(outs)
        
        x_c1, x_c2, x_c3, x_c4 = outs  
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        
        return x

    def get_latent(self, x):
        return self.encoder_vq.encode(x).mode()
