import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable

import math
from torch.nn.functional import linear, normalize ###

from typing import Optional

import torch
from torch import Tensor, nn

from ripser import Rips
from persim import PersImage
from persim import PersistenceImager
import matplotlib.pyplot as plt
from ripser import ripser
import numpy as np
from tsc import signal_persistence, compress_tsc, reconstruct_tsc
from tsc.utils.viz import plot_persistence
import random

from backbones.persistent_homology import *

class Embedding(nn.Embedding):  

    def __init__(
        self,
        #opts,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        *args,
        **kwargs
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight[self.padding_idx], 0)

            

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VITBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        return self.bn(x)


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,          ## multi-head attention
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  ## kv
        self.q_ = nn.Linear(dim, dim * 1, bias=qkv_bias)   ## q
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.map_1d = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)

    def forward(self, x, topo_all_fea, block_pos):

        if block_pos != "end":  
            x_topo_information = x
            
        elif block_pos == "end":     

            x_topo_information = x + topo_all_fea
         
        with torch.cuda.amp.autocast(True):
            batch_size, num_token, embed_dim = x.shape  
            
            kv = self.kv(x_topo_information).reshape(
                batch_size, num_token, 2, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
            q_ = self.q_(x).reshape(
                batch_size, num_token, 1, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)  
            
        with torch.cuda.amp.autocast(False):
                        
            q, k, v = q_[0].float(), kv[0].float(), kv[1].float()  
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)   
        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 num_patches: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Callable = nn.ReLU6,
                 norm_layer: str = "ln", 
                 patch_n: int = 144):
        super().__init__()

        if norm_layer == "bn":
            self.norm1 = VITBatchNorm(num_features=num_patches)
            self.norm2 = VITBatchNorm(num_features=num_patches)
        elif norm_layer == "ln":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.extra_gflops = (num_heads * patch_n * (dim//num_heads)*patch_n * 2) / (1000**3)

    def forward(self, x, topo_all_fea, block_pos):

        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(topo_all_fea), block_pos ))  

                               
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=108, patch_size=9, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        #self.proj = nn.Conv2d(in_channels, embed_dim,              
                              #kernel_size=patch_size, stride=patch_size)     
        conv_kernel_size = 32
        vocab_size = 256
        embed_dim = 512
        
        self.embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=128
        ) 
        
        nn.init.trunc_normal_(self.embeddings.weight, std=math.sqrt(1.0 / embed_dim))
        
        
        self.token_reduction_net = nn.Sequential(
        nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=32,
                stride=16,
                bias=False,
            ),
            
        nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=63,
                stride=16,
                bias=False,
            )   
        ) 
        
        self.topo_increase = nn.Sequential(
            nn.Linear(in_features=19, out_features=144, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=144, out_features=144*512, bias=False),
            nn.ReLU()
        ).cuda()
        
        self.topo_conv=nn.Conv1d(in_channels=512, out_channels=3, kernel_size=30, stride=6).cuda()
        
    def forward(self, x, phase="train"):
        
        if phase == "train":
            
            x_numpy = x.cpu().numpy()
        
            compress_array = np.zeros((x.shape[0], 144), dtype=np.float64)
        
            compress_bytes = torch.zeros(x.shape[0], 144).cuda()   
        
            zero_agent = np.zeros((144), dtype=np.float64)
        
            probability = 0.3
        
            for index_ in range(x_numpy.shape[0]):
                if random.random() <= probability:
                    reshaped_x = np.column_stack((np.arange(x.size()[1]), x_numpy[index_]))  
                    data_all_pers_values = compress_tsc(reshaped_x, num_indices_to_keep=144)  
                    compress_array[index_] = data_all_pers_values[:, 1]  
                    norm = np.linalg.norm(compress_array[index_])
                    compress_array[index_] = compress_array[index_] / norm
                else:
                    compress_array[index_] = zero_agent   
            
        
            compress_bytes = torch.tensor(compress_array).float().cuda()  
        
        x = self.embeddings(x.int()) 
        
        x = self.token_reduction_net(x.permute(0, 2, 1)).permute(0, 2, 1)  
        
        x_topo = self.topo_conv(x.permute(0, 2, 1)).permute(0, 2, 1) 

        topo_information = []
        for index in range(x.size()[0]):
            pd_value = compute_pd(x_topo[index])  #

            topo_information.append(pd_value)   
            
        topo_feature = torch.stack(topo_information).cuda()   
        
        topo_all_fea_before = self.topo_increase(topo_feature).reshape(x.shape[0],144,512).cuda()
        
        topo_all_fea = topo_all_fea_before
        
        if phase == "train": 
            x = compress_bytes.unsqueeze(2) + x    
        
        return x, topo_all_fea
        


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size: int = 112,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 hybrid_backbone: Optional[None] = None,
                 norm_layer: str = "ln",
                 mask_ratio = 0.1,
                 using_checkpoint = False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        
        if hybrid_backbone is not None:
            raise ValueError
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.mask_ratio = mask_ratio
        self.using_checkpoint = using_checkpoint
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        patch_n = (img_size//patch_size)**2
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                      num_patches=num_patches, patch_n=patch_n)
                for i in range(depth)]
        )        
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops

        if norm_layer == "ln":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_layer == "bn":
            self.norm = VITBatchNorm(self.num_patches)

        # features head
        self.feature = nn.Sequential(
            nn.Linear(in_features=embed_dim * num_patches, out_features=embed_dim, bias=False),
            nn.BatchNorm1d(num_features=embed_dim, eps=2e-5),
            nn.Linear(in_features=embed_dim, out_features=num_classes, bias=False),
            nn.BatchNorm1d(num_features=num_classes, eps=2e-5)
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
        ## SEModule FC
        self.senet = nn.Sequential(
            nn.Linear(in_features=embed_dim * num_patches, out_features=num_patches, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=num_patches, out_features=num_patches, bias=False),
            nn.Sigmoid()
        )
                    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def random_masking(self, x, topo_all_fea, mask_ratio=0.1):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.size()  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        topo_all_fea_masked = torch.gather(
            topo_all_fea, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, topo_all_fea_masked, mask, ids_restore

    def forward_features(self, x, phase="train"):
        B = x.shape[0]
        
        if phase=="train":
            x, topo_all_fea = self.patch_embed(x, phase="train")
        else:
            x, topo_all_fea = self.patch_embed(x, phase="infer")
        
        x = x + self.pos_embed   
        x = self.pos_drop(x)
        

        if self.training and self.mask_ratio > 0:

            x, topo_all_fea, _, ids_restore = self.random_masking(x, topo_all_fea)
          
        for func in self.blocks:                
            if self.using_checkpoint and self.training:   ## train
            
                from torch.utils.checkpoint import checkpoint
                
                if func == self.blocks[-1]:   
                    x = checkpoint(func, x, topo_all_fea, "end") 
                else:
                    x = checkpoint(func, x, topo_all_fea, "e") 
                
            else:
                if func == self.blocks[-1]:   
                    x = func(x, topo_all_fea, block_pos="end")  
       
                else:
                    x = func(x, topo_all_fea, block_pos="e") 

                    
        x = self.norm(x.float())                 
        
        if self.training and self.mask_ratio > 0:

            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = x_
            
        orginal = x
        out = torch.reshape(x, (B, self.num_patches * self.embed_dim))
        out = self.senet(out)
        out_softmax = out.softmax(dim=1)
        out = torch.reshape(out, (B, self.num_patches, 1))
        out = out * orginal
        
        return torch.reshape(out, (B, self.num_patches * self.embed_dim))
      
    def forward(self, x, phase="train"):
        if phase == "train":
            x = self.forward_features(x, phase="train")
        else:
            x = self.forward_features(x, phase="infer")
            
        out_x = torch.reshape(x, (x.shape[0], self.num_patches, self.embed_dim) )
        patch_std = torch.std(out_x, dim=2)
            
        x = self.feature(x)   
            
        first_size = x.size()[0]
        weight_ = torch.randn(first_size,144).cuda()

        patch_ = patch_std
            
        bottleneck_embedding = x
            
        return x, weight_, patch_   
            


