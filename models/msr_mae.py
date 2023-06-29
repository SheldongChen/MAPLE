import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *
from transformer import *


def ShuffleIndex(index: list, sample_ratio: float):
    if len(index) < 4:
        raise ValueError("ipnuts must be more than 4")
    else:
        sample_length = int(sample_ratio * len(index))

        sample_list = random.sample(index, sample_length)
        mask_list = [x for x in index if x not in sample_list]  # get the remain index
        # print(len(index),len(sample_list))
        assert len(sample_list) == int(len(index) * sample_ratio), "sample length must be same as the ratio!!!"
    return sorted(sample_list), sorted(mask_list)


def MaskEmbeeding(token_emb, mask_ratio):
    """get the mask embeeding after patch_emb + pos_emb
    """
    # token_emb: [Bs, Length, Channels]
    token_length = token_emb.shape[1]
    token_index = list(range(0, token_length))
    # print(len(token_index))
    sample_index, mask_index = ShuffleIndex(token_index, sample_ratio=1-mask_ratio)

    x = token_emb[:, sample_index, :]
    return x, sample_index, mask_index


class UnMaskEmbeeding(nn.Module):
    """get the mask embeeding from the image -> 127 to embeeding, before the position embeeding
    """

    def __init__(self, dim_decoder):
        super().__init__()
        self.dim_decoder = dim_decoder
        # used for mask images
        self.raw_inputs = torch.ones(size=(1, 1, dim_decoder))
        self.proj = nn.Linear(dim_decoder, dim_decoder, bias=True)

    def forward(self, x, sample_index, mask_index):
        b, l, c = x.shape  # [B, L*mask_ratio, c]
        length = len(sample_index) + len(mask_index)
        assert l < length
        assert self.dim_decoder==c
        # force keeping self.raw_inputs.requires_grad == False
        self.raw_inputs.requires_grad = False
        self.raw_inputs = self.raw_inputs.to(x.device)
        decoder_embeeding = nn.Parameter(torch.zeros((b, length, self.dim_decoder))).to(x.device) #b,L,c

        patch_embeeding = self.proj(self.raw_inputs)  # 1,1,c

        decoder_embeeding[:, sample_index, :] = x
        decoder_embeeding[:, mask_index, :] = patch_embeeding

        return decoder_embeeding

class PositionEmbed(nn.Module):
    def __init__(self, num_patches=12, d_model=512, num_tokens=0):
        super().__init__()
        import math
        # Compute the positional encodings once in log space.
        self.num_tokens = num_tokens
        assert self.num_tokens >=0, "num_tokens must be class token or no, so must 0 or 1"
        pe = torch.zeros(num_patches+self.num_tokens, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, num_patches + self.num_tokens).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        #pe = pe.cuda()
        self.register_buffer('pe', pe)

    def __call__(self):
        return self.pe

class P4Transformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth_xyz, depth_t, heads, dim_head,                              # transformer
                 mlp_dim, num_classes, only_cls=False, depth_t_decoder=8, dim_decoder=512, mask_ratio=0.5):              # output
        super().__init__()
        self.only_cls = only_cls

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer_1 = Transformer(dim, depth_xyz, heads, dim_head, mlp_dim)
        self.transformer_2 = Transformer(dim, depth_t, heads, dim_head, mlp_dim)

        if not only_cls:
            self.transformer_decoder_3 = Transformer(dim_decoder, depth_t_decoder, heads, dim_head, mlp_dim)
            # project encoder embeeding to decoder embeeding
            self.proj = nn.Linear(dim, dim_decoder)
            self.restruction = nn.Linear(dim_decoder, dim)
            self.mask_ratio = mask_ratio
            self.unmask_embed = UnMaskEmbeeding(dim_decoder)
            self.norm_encoder = nn.LayerNorm(dim)
            self.decoder_pos_embedding = PositionEmbed(12, dim_decoder, 0)()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def low_feature_forward(self, input_clips):
        device = input_clips.get_device()
        xyzs, features = self.tube_embedding(input_clips)                                                                                         # [B, L, n, 3], [B, L, C, n]

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)

        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0]*xyzts.shape[1], xyzts.shape[2], xyzts.shape[3]))                           # [B*L, n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        B, L, n, C = features.shape

        features = torch.reshape(input=features, shape=(B*L, n, C))         # [B*L, n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        return xyzts, features, [B, L, n, C]

    def forward(self, input_clips, forward_type='cls'): # input: [B, L, N, 3]
        assert forward_type in ['cls', 'mask', 'both']

        # xyzts:position enbedding,     [B*L, n, 4]
        # features:feature of 3d cloud, [B*L, n, C]
        xyzts, features, shapes = self.low_feature_forward(input_clips)
        B, L, n, C = shapes

        embedding = xyzts + features
        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        # generate embedding of xyz
        embedding_xyz = self.transformer_1(embedding)
        embedding_xyz = torch.reshape(input=embedding_xyz, shape=(B, L, n, C))
        embedding_xyz = torch.max(input=embedding_xyz, dim=-2, keepdim=False, out=None)[0]   # [B, L, C]
        device_gpu =  embedding_xyz.device
        if forward_type == 'cls':
            # generate embedding of t
            embedding_t = self.transformer_2(embedding_xyz)  # [B, L, C]

            output_cls = torch.max(input=embedding_t, dim=1, keepdim=False, out=None)[0]  # [B, C]
            # generate class
            output_cls = self.mlp_head(output_cls)  # [B, num_class]
            return output_cls

        elif not self.only_cls and forward_type == 'mask':
            embedding_xyz_clone = embedding_xyz.clone() # [B, L, C]
            # mask the patchemb&posemb
            # embedding_xyz = embedding_xyz.detach() + 0.0 * embedding_xyz
            mask_embedding_xyz, sample_index, mask_index = MaskEmbeeding(embedding_xyz, mask_ratio=self.mask_ratio)
            mask_embedding_t = self.transformer_2(mask_embedding_xyz)  # [B, L*mask_ratio, C]

            # for proj
            norm_embedding_t = self.norm_encoder(mask_embedding_t)
            mask_proj_embedding_t = self.proj(norm_embedding_t) # [B, L*mask_ratio, cc]

            # for decoder
            assert self.decoder_pos_embedding.requires_grad == False
            proj_embedding_t = self.unmask_embed(mask_proj_embedding_t, sample_index, mask_index)  # [B, L, cc]
            proj_embedding_t = proj_embedding_t + self.decoder_pos_embedding.to(proj_embedding_t.device)
            proj_embedding_t = self.transformer_decoder_3(proj_embedding_t) # [B, L, cc]
            restruct_embedding_xyz = self.restruction(proj_embedding_t) # [B, L, C]

            return restruct_embedding_xyz, embedding_xyz_clone, mask_index

        elif not self.only_cls and forward_type == 'both':
            embedding_xyz_clone = embedding_xyz.clone()
            # mask the patchemb&posemb
            mask_embedding_xyz, sample_index, mask_index = MaskEmbeeding(embedding_xyz, mask_ratio=self.mask_ratio)
            mask_embedding_t = self.transformer_2(mask_embedding_xyz)  # [B, L*mask_ratio, C]

            # for cls
            output_cls = torch.max(input=mask_embedding_t, dim=1, keepdim=False, out=None)[0]  # [B, C]
            # generate class
            output_cls = self.mlp_head(output_cls)  # [B, num_class]

            # for proj
            norm_embedding_t = self.norm_encoder(mask_embedding_t)
            mask_proj_embedding_t = self.proj(norm_embedding_t) # [B, L*mask_ratio, cc]

            # for decoder
            assert self.decoder_pos_embedding.requires_grad == False
            proj_embedding_t = self.unmask_embed(mask_proj_embedding_t, sample_index, mask_index)  # [B, L, cc]
            proj_embedding_t = proj_embedding_t + self.decoder_pos_embedding.to(proj_embedding_t.device)
            proj_embedding_t = self.transformer_decoder_3(proj_embedding_t) # [B, L, cc]
            restruct_embedding_xyz = self.restruction(proj_embedding_t) # [B, L, C]

            return restruct_embedding_xyz, embedding_xyz_clone, mask_index, output_cls
        else:
            assert False, "forward_type must in ['cls', 'mask', 'both']"
