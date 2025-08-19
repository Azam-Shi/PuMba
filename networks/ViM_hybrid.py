"""
# Reference:
# https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py
"""

import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
import math
import ml_collections
from collections import namedtuple
from utils.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from utils.rope import *
import random

try:
    from utils.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def get_ml_config(params):
    """Returns the ViT configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (params['patch_size'], params['patch_size'])})

    config.hidden_size = params['hidden_size']
    config.depth = params['depth']
    config.classifier = 'token'
    config.representation_size = None
    # Additional model params
    config.update({
        'drop_path_rate': 0.1,
        'rms_norm': True,
        'norm_epsilon': 1e-5,
        'fused_add_norm': True,
        'residual_in_fp32': True,
        'if_cls_token': True,
        'stride': 4,
        'if_bidirectional': True,
        'if_abs_pos_embed': True,
        'if_rope': False,
        'if_rope_residual': False,
        'flip_img_sequences_ratio': -1.0,
        'use_double_cls_token': False,
        'use_middle_cls_token': True
    })
    return config

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self,
                 config,
                 img_size=32,
                 patch_size=4,
                 stride=4,
                 depth=8,
                 embed_dim=16,
                 d_state=8,
                 ssm_cfg=None,
                 channels=13,
                 drop_path=0.,
                 residual_in_fp32=False,
                 drop_path_rate=0.1,
                 norm_epsilon=1e-5,
                 fused_add_norm=True,
                 layer_idx=None,
                 device=None,
                 dtype=None,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_divide_out=False,
                 rms_norm=True,
                 if_bidirectional=True,
                 if_cls_token=True,
                 **kwargs):
        super(Encoder, self).__init__()

        self.if_cls_token = if_cls_token
        self.embed_dim = embed_dim
        self.fused_add_norm = True
        self.residual_in_fp32 = True
        self.drop_path_rate = drop_path_rate
        self.if_bidirectional = if_bidirectional

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr

        if if_bimamba:
            bimamba_type = "v1"
        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        # mixer_cls = partial(Mamba, d_state=d_state, use_fast_path=True, bimamba_type="v2")
        mixer_cls = partial(Mamba, d_state=d_state, use_fast_path=True, bimamba_type="v2", layer_idx=layer_idx, **ssm_cfg, **factory_kwargs) 
        
       
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )

        self.layers = nn.ModuleList([
            Block(
                dim=embed_dim,
                mixer_cls=mixer_cls,
                norm_cls=norm_cls,
                drop_path=inter_dpr[i],
                fused_add_norm=self.fused_add_norm,
                residual_in_fp32=self.residual_in_fp32
            ) for i in range(depth)
        ])
        
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    
    
    def forward(self, x):
        residual = None
        hidden_states = x

        if not self.if_bidirectional:
            for layer in self.layers:
                hidden_states, residual = layer(hidden_states, residual)
        else:
            for i in range(len(self.layers) // 2):
                hidden_states_f, residual_f = self.layers[i * 2](hidden_states, residual)
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1])
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32
            )
        return hidden_states

     
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            self.gradients = None
    
    def save_gradients(self, attn_gradients):
        self.gradients = attn_gradients

    def get_gradients(self):
        return self.gradients
    
    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                # residual = residual + self.drop_path(hidden_states)
                residual = (hidden_states + residual)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
 
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        if residual.requires_grad:
            residual.register_hook(self.save_gradients)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    d_state=8,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, d_state=d_state, use_fast_path=True, bimamba_type="v2", layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02, 
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1, 
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class VisionMamba(nn.Module):
    def __init__(self, config, img_size=32, patch_size=4, stride=4, embed_dim=16, depth=8, d_state=8, rms_norm=True, initializer_cfg=None, fused_add_norm=True, residual_in_fp32=True, norm_epsilon=1e-5,
                 use_double_cls_token=False, use_middle_cls_token=True, channels=13, num_classes=2, drop_path_rate=0.1):
        super().__init__()
        self.if_bidirectional = config.get("if_bidirectional", True)
        self.if_abs_pos_embed = config.get("if_abs_pos_embed", True)
        self.if_rope = config.get("if_rope", False)
        self.if_rope_residual = config.get("if_rope_residual", False)
        self.flip_img_sequences_ratio = config.get("flip_img_sequences_ratio", -1.0)
        self.if_cls_token = config.get("if_cls_token", True)
        self.use_double_cls_token = config.get("use_double_cls_token", False)
        self.use_middle_cls_token = config.get("use_middle_cls_token", True)
        self.embed_dim = embed_dim
        self.num_features = self.embed_dim
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.num_tokens = 2 if self.use_double_cls_token else (1 if self.if_cls_token else 0)
 
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if self.if_cls_token:
            if self.use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                
        if self.if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=config.get("drop_rate", 0.0))

        if self.if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type="v2",
                    drop_path=inter_dpr[i],
                    if_divide_out=True,
                    init_layer_scale=None,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon)

        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if self.if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if self.if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)
        
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    
    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        x = self.patch_embed(x)
        B = x.shape[0]
        B, N, D = x.shape  

        if self.if_cls_token:
            cls_token = self.cls_token.expand(B, 1, D)
            half = N // 2
            x = torch.cat((x[:, :half], cls_token, x[:, half:]), dim=1)  # CLS in center

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)
        
        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True
            
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])
                    
                    if self.if_rope:
                        hidden_states = self.rope(hidden_states)
                        if residual is not None and self.if_rope_residual:
                            residual = self.rope(residual)
                    
                    if if_flip_img_sequences and self.if_rope:
                        hidden_states = hidden_states.flip([1])
                        if residual is not None:
                            residual = residual.flip([1]) 
                    
                    hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                    )
        else:
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])
            
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )          
        if self.if_cls_token:
            return hidden_states[:, half, :]  
        else:
            raise NotImplementedError
    
    def forward(self, x, return_features=False, inference_params=None):
        features = self.forward_features(x, inference_params)
        return features
    

def vision_mamba(config: ml_collections.ConfigDict, channels: int, num_classes: int = 2) -> VisionMamba:
    return VisionMamba(
        config=config,
        img_size=config.get("img_size", 32),
        patch_size=config.patches.size[0],
        stride=config.get("stride", 4),
        embed_dim=config.hidden_size,
        depth=config.depth,
        d_state=config.get("d_state", 8),
        channels=channels,
        num_classes=num_classes,
        drop_path_rate=config.get("drop_path_rate", 0.1)
    ) 
class ViM_Hybrid_encoder(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict, n_individual: int, img_size: int = 32,
                 num_classes: int = 2, zero_head: bool = False, vis: bool = False, channels: int = 13):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.forward_features = vision_mamba(config, channels, num_classes)
        self.individual_nn = nn.Linear(n_individual, n_individual)
        self.combine_nn = nn.Linear(config.hidden_size + n_individual, config.hidden_size)
        self.af_ind = nn.GELU()
        self.af_combine = nn.GELU()

    def forward(self, x, individual_feat, return_features=False, inference_params=None):
        x = self.forward_features(x, inference_params=inference_params)        
        individual_x = self.individual_nn(individual_feat)
        individual_x = self.af_ind(individual_x)
        x = torch.cat([x, individual_x], dim=1)  
        x = self.combine_nn(x)  
        x = self.af_combine(x)  
        return x, None
