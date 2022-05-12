# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import bmtrain as bmt
from model_center.layer import LayerNorm
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
try:
    from torch import _assert
except ImportError:
    def _assert(condition:bool, message:str):
        assert condition, message

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class Identity(bmt.DistributedModule):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
    
    def forward(self, input):
        return input


class Conv2d(bmt.DistributedModule):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                dtype=torch.float,
                int8: bool=False,
                init_mean : float=0.0,
                init_std : float = 1,
                bias : bool=True,
                padding_mode='zeros',
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.transposed = None
        self.output_padding = None

        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.padding_mode = padding_mode

        kernel = to_2tuple(kernel_size)
        self.weight = bmt.DistributedParameter(
            torch.empty((out_channels, int(in_channels/groups), kernel[0], kernel[1]), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.bias = bmt.DistributedParameter(
            torch.empty((out_channels,), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.zeros_)
        ) if bias else None
        self.int8=int8

    def forward(self, x : torch.Tensor):
        x = F.conv2d(x,
                    weight=self.weight, 
                    bias=self.bias, 
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    )
        
        return x
        

class Linear(bmt.DistributedModule):
    r"""A fully connected layer, which performs :math:`\pmb{y} = \mathbf{W} \pmb{x} + \pmb{b}`
    Args:
        dim_in (int): input dimension of :math:`\pmb{x}`
        dim_out (int): output dimension of :math:`\pmb{y}`
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 1.
        bias (bool, optional): whether to add bias term :math:`\pmb{b}`. Defaults to False.
    """
    def __init__(self,
                 in_features : int,
                 out_features : int,
                 length_scale : bool = False,
                 length_scale_before : bool = False,
                 dtype = torch.float,
                 int8 : bool = False,
                 init_mean : float = 0.0,
                 init_std : float = 1,
                 bias : bool = True,
                ):
        super().__init__()
        self.in_features = in_features
        self.weight = bmt.DistributedParameter(
            torch.empty((out_features, in_features), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.bias = bmt.DistributedParameter(
            torch.empty((out_features,), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.zeros_)
        ) if bias else None
        self.length_scale = length_scale
        self.length_scale_before = length_scale_before
        self.int8 = int8

    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """
        if self.length_scale and self.length_scale_before:
            x = x / math.sqrt(self.in_features)
        x = F.linear(x, self.weight)
        if self.length_scale and not self.length_scale_before:
            x = x / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class PatchEmbed(bmt.DistributedModule):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,
                img_size=224,
                patch_size=16,
                in_chans=3,
                embed_dim=768,
                norm_layer=None,
                flatten=True,
                dtype=torch.half
                ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, dtype=dtype)
        self.norm = norm_layer(embed_dim) if norm_layer else Identity

    def forward(self, x):
        B,C,H,W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        # x = self.norm(x)
        return x


class Mlp(bmt.DistributedModule):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                in_features, 
                hidden_features=None, 
                out_features=None, 
                act_layer=torch.nn.functional.gelu, 
                drop=0.0,
                dtype=torch.half
                ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, dtype=dtype)
        self.act = act_layer
        self.fc2 = Linear(hidden_features, out_features, dtype=dtype)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(bmt.DistributedModule):
    def __init__(self, 
                dim, 
                num_heads=8, 
                qkv_bias=False, 
                qk_scale=None, 
                attn_drop=0., 
                proj_drop=0., 
                length_scale=False,
                dtype=torch.float,
                int8=False,
                init_mean : float = 0.0,
                init_std : float = 1,
                bias : bool = True,
                ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
        self.qkv = Linear(
            in_features = dim,
            out_features = dim*3,
            bias=qkv_bias,
            length_scale=length_scale,
            length_scale_before=False,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
        )

        self.proj = Linear(
            in_features = dim,
            out_features = dim,
            bias=qkv_bias,
            length_scale=length_scale,
            length_scale_before=False,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
        )

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(bmt.DistributedModule):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=torch.nn.functional.gelu, norm_layer=LayerNorm,dtype=torch.float):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,dtype=dtype)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,dtype=dtype)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class VisionTransformer(bmt.DistributedModule):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, dtype=torch.float):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (bmt.DistributedModule): normalization layer
            dtype: Defaults to torch.float.

        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or LayerNorm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, dtype=dtype)
        num_patches = self.patch_embed.num_patches

        self.cls_token = bmt.DistributedParameter(torch.empty((1,1,embed_dim), dtype=dtype))
        self.pos_embed = bmt.DistributedParameter(torch.empty((1,num_patches+1,embed_dim), dtype=dtype))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks =  bmt.TransformerBlockList(
                        [
                            bmt.CheckpointBlock(
                                Block(
                                      dim=embed_dim, num_heads=num_heads, 
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                      qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate, 
                                      drop_path=dpr[i], norm_layer=norm_layer, 
                                      dtype=dtype
                                    )
                                ) for i in range(depth)
                            ]
                    )
        self.norm = norm_layer(embed_dim)
        self.head = Linear(embed_dim, num_classes, dtype=dtype)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, 1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x[:,0])
        return x



def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint


if __name__ == '__main__':
    import bmtrain as bmt
    from functools import partial
    from model_center.layer import LayerNorm
    bmt.init_distributed(seed=0)
    vit = VisionTransformer(img_size=256, patch_size=16, 
                            embed_dim=768, depth=12, num_heads=12, 
                            mlp_ratio=4, qkv_bias=True, 
                            norm_layer=partial(LayerNorm,dtype=torch.float,eps=1e-6), 
                            dtype=torch.float
                        )
    checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
    state_dict = checkpoint["model"]
    pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], vit)
    state_dict['pos_embed'] = pos_embed_reshaped
    state_dict = bmt.store.DistributedStateDictWrapper(state_dict)
    msg = vit.load_state_dict(state_dict,strict=False)


