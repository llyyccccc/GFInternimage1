import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import numpy as np
import torch.nn as nn

import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from .modules.EFF import EFF
import math
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import math
from timm.models.layers import DropPath, trunc_normal_

from IPython import embed

from .modules.ELA import ELA
from .modules.EMCAD import EUCB
from .modules.EMA import EMA
from .modules.MEEM import MEEM

# from modules.YCA import EMA
from .modules.DMC import DMC_fusion

from .modules.SCSA import SCSA
from .modules.PAM import PAM
from .modules.EMAb import EMAb
from .modules.EFC import EFC
from .modules.FEM import FEM
from .modules.FF import FreqFusion
from .modules.BFAM import BFAM
from .modules.AIFI import AIFI
from .modules.DLK import DLK
from .modules.MCFS import MCFS


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
# class SeparableConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
#         super(SeparableConv2d, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
#                                bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pointwise(x)
#         return x






class Mlp(nn.Module):
    """Mlp implemented by with 1*1 convolutions.
    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 #  act_cfg=dict(type='GELU'),
                 act_layer=nn.GELU,
                 drop_path=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        #self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        # self.act = build_activation_layer(act_cfg)
        self.act = act_layer()
        self.dwconv = DWConv(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop_path)
        # self.sconv = SeparableConv2d(hidden_features,hidden_features)
    
    
   
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.sconv(x)
        # print('以应用')
        x = self.fc2(x)
        x = self.drop(x)

        x = x.flatten(2).transpose(1, 2)
        return x



class ELA(nn.Module):
    def __init__(self, dim1, phi):
        super(ELA, self).__init__()
        '''
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。
        '''
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': dim1, 'B': dim1, 'S': dim1 // 8, 'L': dim1 // 8}[phi]
        num_groups = {'T': 16, 'B': 16, 'S': 16, 'L': 16}[phi]
        pad = Kernel_size // 2
        self.con1 = nn.Conv1d(dim1, dim1, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        # print(dim2,num_groups)
        self.GN = nn.GroupNorm(num_groups, dim1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        x_h = self.con1(x_h)  # [b,c,h]
        x_w = self.con1(x_w)  # [b,c,w]
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]
        return x_h * x_w * input

# class ChannelReductionAttention(nn.Module):
#     def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
#         super().__init__()
#         assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."
#
#         self.dim1 = dim1//2
#         self.pool_ratio = pool_ratio
#         self.num_heads = num_heads
#         head_dim = dim1 // num_heads
#
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim1, dim1)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
#         self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
#         self.norm = nn.LayerNorm(dim1)
#         self.act = nn.GELU()
#         self.apply(self._init_weights)
#
#         # 整合HiLo中lofi相关参数和层定义
#         self.l_heads = int(num_heads )  # 这里假设一半头用于类似Lo-Fi机制，可根据实际调整
#         self.l_dim = self.l_heads * head_dim
#         if self.l_heads > 0:
#             self.l_q = nn.Linear(dim1//2, self.l_dim, bias=qkv_bias)
#             self.l_kv = nn.Linear(dim1//2, self.l_dim * 2, bias=qkv_bias)
#             self.l_proj = nn.Linear(self.l_dim, self.l_dim)
#             self.sr_lofi = nn.AvgPool2d(kernel_size=2, stride=2)  # 假设窗口大小为2，可按需调整
#         self.conv3 = nn.Conv2d(self.l_dim, dim1, 1)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, h, w):
#
#         x, x2 = x.chunk(2, dim=1)
#         B, N, C = x.shape
#         # 原有的常规注意力机制相关操作开始
#
#         q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#         x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
#         x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
#         x_ = self.norm(x_)
#         x_ = self.act(x_)
#
#         k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#         v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         # 原有的常规注意力机制相关操作结束
#
#         # 融入类似lofi的逻辑处理
#         if self.l_heads > 0:
#             # 调整x维度便于后续lofi操作，这里将x从[B, N, C]变为[B, h, w, C]格式（假设h、w是特征图的高、宽）
#             x_lofi = x2.reshape(B, h, w, C)
#
#
#
#             # 生成查询向量q_lofi，维度调整符合多头注意力机制要求
#             q_lofi = self.l_q(x_lofi).reshape(B, h * w, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
#
#             if self.sr_lofi.kernel_size > 1:
#                 # 对特征图进行池化等操作生成键值对（类似HiLo中lofi做法）
#                 x_lofi_ = x_lofi.permute(0, 2, 3, 1)
#                 x_lofi_ = self.sr_lofi(x_lofi_).reshape(B, C, -1).permute(0, 2, 1)
#                 kv_lofi = self.l_kv(x_lofi_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
#             else:
#                 kv_lofi = self.l_kv(x_lofi).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
#             k_lofi, v_lofi = kv_lofi[0], kv_lofi[1]
#
#             # 计算lofi的注意力权重
#             attn_lofi = (q_lofi @ k_lofi.transpose(-2, -1)) * self.scale
#             attn_lofi = attn_lofi.softmax(dim=-1)
#
#             # 根据注意力权重对值向量加权求和并调整维度，恢复特征图维度格式
#             x_lofi = (attn_lofi @ v_lofi).transpose(1, 2).reshape(B, N, self.l_dim)
#             x_lofi = self.l_proj(x_lofi)
#
#             # 将原注意力输出和类似lofi的注意力输出进行融合（这里简单拼接，可根据实际调整融合方式）
#
#         # x = x_lofi +x
#         x = torch.cat([x, x_lofi.reshape(B, N, self.l_dim)], dim=-1)
#
#
#         return x

# class ChannelReductionAttention(nn.Module):
#     def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
#         super().__init__()
#         assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."
#
#         self.dim1 = dim1
#         self.pool_ratio = pool_ratio
#         self.num_heads = num_heads
#         head_dim = dim1 // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.scales = [3, 5]  # 定义多尺度的卷积核大小列表
#         self.conv_layers = nn.ModuleList([
#             nn.Conv2d(dim1, dim1, kernel_size=scale, padding=(scale - 1) // 2, groups=dim1) for scale in self.scales
#         ])  # 为每个尺度创建卷积层，这里使用了保持空间尺寸不变的填充方式
#         self.q_layers = nn.ModuleList([nn.Linear(dim1, self.num_heads, bias=qkv_bias) for _ in self.scales])
#         self.k_layers = nn.ModuleList([nn.Linear(dim1, self.num_heads, bias=qkv_bias) for _ in self.scales])
#         self.v_layers = nn.ModuleList([nn.Linear(dim1, dim1, bias=qkv_bias) for _ in self.scales])
#
#         # 用于原始输入的q、k、v生成模块
#         self.original_q_layer = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.original_k_layer = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.original_v_layer = nn.Linear(dim1, dim1, bias=qkv_bias)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim1, dim1)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
#         self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
#         self.norm = nn.LayerNorm(dim1)
#         self.act = nn.GELU()
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, h, w):
#         B, N, C = x.shape
#         # 处理原始输入的qkv生成和注意力计算
#         q_original = self.original_q_layer(x).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#         x_original = x.permute(0, 2, 1).reshape(B, C, h, w)
#         x_original = self.sr(self.pool(x_original)).reshape(B, C, -1).permute(0, 2, 1)
#         x_original = self.norm(x_original)
#         x_original = self.act(x_original)
#
#         k_original = self.original_k_layer(x_original).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#         v_original = self.original_v_layer(x_original).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2,
#                                                                                                                    1, 3)
#
#         attn_original = (q_original @ k_original.transpose(-2, -1)) * self.scale
#         attn_original = attn_original.softmax(dim=-1)
#         attn_original = self.attn_drop(attn_original)
#
#         x_attn_original = (attn_original @ v_original).transpose(1, 2).reshape(B, N, C)
#
#         multi_scale_attn_results = []
#         for i in range(len(self.scales)):
#             conv_layer = self.conv_layers[i]
#             q_layer = self.q_layers[i]
#             k_layer = self.k_layers[i]
#             v_layer = self.v_layers[i]
#
#             # 获取当前尺度的特征
#             scale_x = conv_layer(x.permute(0, 2, 1).reshape(B, C, h, w)).reshape(B, C, -1).permute(0, 2, 1)
#
#             q = q_layer(scale_x).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#             x_ = scale_x.permute(0, 2, 1).reshape(B, C, h, w)
#             x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             x_ = self.act(x_)
#
#             k = k_layer(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#             v = v_layer(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#
#             x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
#             multi_scale_attn_results.append(x_attn)
#
#         # 融合原始输入和多尺度的注意力结果
#         final_result = x_attn_original
#         for attn_result in multi_scale_attn_results:
#             final_result += attn_result
#         final_result = self.proj(final_result)
#         final_result = self.proj_drop(final_result)
#         return final_result

# class ChannelReductionAttention(nn.Module):
#     def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
#         super().__init__()
#         assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."
#
#         self.dim1 = dim1
#         # self.dim2 = dim2
#         self.pool_ratio = pool_ratio
#         self.num_heads = num_heads
#         head_dim = dim1 // num_heads
#
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim1, dim1)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
#         # self.pool = F.adaptive_avg_pool2d(pool_ratio, pool_ratio)
#         self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
#         self.norm = nn.LayerNorm(dim1)
#         self.act = nn.GELU()
#         self.apply(self._init_weights)
#         self.dvc = DCNv4(dim1)
#
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, h, w):
#         B, N, C = x.shape
#
#         q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#
#         c = self.dvc(x)
#         c = self.norm(c)
#         c = self.act(c)
#
#         k1 = self.k(c).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#         v1 = self.v(c).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#
#         attn1 = (q @ k1.transpose(-2, -1)) * self.scale
#         # attn1 = attn1.softmax(dim=-1)
#         # attn1 = self.attn_drop(attn1)
#         # x = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
#
#         x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
#
#         x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
#         # x_ = self.pool(self.act(self.sr(x_))).reshape(B, C, -1).permute(0, 2, 1)
#
#
#         x_ = self.norm(x_)
#
#         x_ = self.act(x_)
#
#
#
#         k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#
#         v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         # print(v.size())
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         #加入可变形卷积
#
#         attn2 = torch.cat([attn1, attn], dim=-1).softmax(dim=-1)
#
#         attn2 = self.attn_drop(attn2)
#         # total_len = attn2.size(-1)
#         split_size = attn1.size(-1)
#         attn1, attn = attn2.split(split_size, dim=-1)
#
#         # attn = attn.softmax(dim=-1)
#         # attn = self.attn_drop(attn)
#         # attn1 = self.attn_drop(attn1)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         attn1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
#
#         x = attn1 + x
#         x = self.proj(x)
#         x = self.proj_drop(x)
# #         return x
class ChannelReductionAttention(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        # self.dim2 = dim2
        self.pool_ratio = pool_ratio
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.pool1 = nn.AvgPool2d(1, 1)
        self.pool2 = nn.AvgPool2d(5, 5)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self.apply(self._init_weights)
        # self.glsa = GLSA(dim1, dim1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
# class ChannelReductionAttention(nn.Module):
#     def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
#         super().__init__()
#         assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."
#
#         self.dim1 = dim1
#         # self.dim2 = dim2
#         self.pool_ratio = pool_ratio
#         self.num_heads = num_heads
#         head_dim = dim1 // num_heads
#
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim1, dim1)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
#         self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
#         self.norm = nn.LayerNorm(dim1)
#         self.act = nn.GELU()
#
#         # 新增局部卷积层相关定义
#         self.local_conv = nn.Conv2d(dim1, dim1, kernel_size=3, padding=1, bias=True)  # 定义3x3的局部卷积，填充为1保证尺寸不变
#         self.local_norm = nn.LayerNorm(dim1)  # 局部卷积后的归一化层
#
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, h, w):
#         B, N, C = x.shape
#
#         # 调整维度以便进行卷积操作
#         local_x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
#         local_x_ = self.local_conv(local_x_)  # 进行局部卷积
#         local_x_ = local_x_.reshape(B, -1, h * w).permute(0, 2, 1)  # 恢复维度顺序
#         local_x_ = self.local_norm(local_x_)  # 进行局部卷积后的归一化
#         local_x_ = self.act(local_x_)  # 激活函数处理
#         # 生成查询q，保持原逻辑
#         q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#
#         x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
#         x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
#
#         # 局部卷积操作及处理
#
#         # 全局注意力机制相关操作（保持原逻辑）
#         global_x_ = x_  # 这里为了区分局部和全局处理的特征，复制一份原特征用于全局操作
#         global_x_ = self.act(global_x_)
#
#         k = self.k(global_x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#         v = self.v(global_x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         global_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         global_out = self.proj(global_out)
#         global_out = self.proj_drop(global_out)
#
#         # 特征融合，这里简单地将局部和全局特征相加，你也可以尝试其他融合方式，比如拼接后再接一个线性层等
#         combined_x = local_x_ + global_out
#
#         return combined_x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes*2, out_planes*2,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes*2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




# class global_meta_block(nn.Module):
#
#     def __init__(self, dim1, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
#         super().__init__()
#
#         self.norm1 = norm_layer(dim1)
#         self.norm3 = norm_layer(dim1)
#
#         self.attn = ChannelReductionAttention(dim1=dim1, num_heads=num_heads, pool_ratio=pool_ratio)
#
#         self.ela = ELA(dim1=dim1,phi='T')
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         mlp_hidden_dim = int(dim1 * mlp_ratio)
#         self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_path=drop_path)
#         # self.conv1_1 = BasicConv2d(dim1, dim1, 1)
#         self.conv1_1 = BasicConv2d(dim1, dim1, 1)
#         inner_dim = max(16, dim1 // 4)
#         self.apply(self._init_weights)
#         self.proj = nn.Sequential(
#             nn.Conv2d(dim1 * 2, dim1 * 2, kernel_size=3, padding=1, groups=dim1 * 2),
#             nn.GELU(),
#             nn.BatchNorm2d(dim1 * 2),
#             nn.Conv2d(dim1 * 2, inner_dim, kernel_size=1),
#             nn.GELU(),
#             nn.BatchNorm2d(inner_dim ),
#             nn.Conv2d(inner_dim, dim1 * 2, kernel_size=1),
#             nn.BatchNorm2d(dim1 * 2), )
#
#
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, h, w):
#
#         x1,x2 = x.chunk(2, dim=1)
#         # x1 = self.ela(x1)
#
#         x1 = self.ela(x1)
#         n, _, h, w = x2.shape
#         x2 = x2.flatten(2).transpose(1, 2)
#
#         x2 = self.norm1(x2)
#
#         x2 = self.attn(x2, h, w)
#         x2 = x2.permute(0, 2, 1).reshape(n, -1, h, w)
#         x = torch.cat([x1,x2], dim=1)
#         # x = self.proj(x) + x
#         x = self.conv1_1(x)
#
#         return x


from einops import rearrange, repeat


class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = x_1 * x_2

        x = self.linear2(x)
        # x = self.eca(x)

        return x

class global_meta_block(nn.Module):

    def __init__(self, dim1, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm3 = norm_layer(dim1)

        self.attn = ChannelReductionAttention(dim1=dim1, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_path=drop_path)

        self.apply(self._init_weights)
        self.frfn = FRFN(dim1)
       
        # self.dcv  = DCNv4(dim1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):


        x = x + self.drop_path(self.attn(self.norm1(x), h, w))


        x = x + self.drop_path(self.mlp(self.norm3(x), h, w))
        #x = x + self.drop_path(self.frfn(self.norm3(x)))
        
        
        return x

class EPA2D(nn.Module):
    
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, 
                 qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

       
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)
      
        self.E = nn.Linear(input_size, proj_size)  # H*W -> P
        self.F = nn.Linear(input_size, proj_size)  # H*W -> P

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU()
        )
        self.out_proj2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU()
        )

    def forward(self, x):
        B, N, C = x.shape  # N = H*W (2D)
        
    
        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)  # [4, B, H, N, C/H]
        
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

       
        q_c = q_shared.transpose(-2, -1)  # [B, H, C/H, N]
        k_c = k_shared.transpose(-2, -1)
        v_ca = v_CA.transpose(-2, -1)
        
       
        k_s = self.E(k_shared)           
        v_sa = self.F(v_SA)              

 
        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.temperature
        attn_c = attn_c.softmax(dim=-1)
        attn_c = self.attn_drop(attn_c)
        x_c = (attn_c @ v_ca).permute(0, 2, 1, 3).reshape(B, N, C)

      
        attn_s = (q_shared @ k_s.transpose(-2, -1)) * self.temperature2
        attn_s = attn_s.softmax(dim=-1)
        attn_s = self.attn_drop_2(attn_s)
        x_s = (attn_s @ v_sa).permute(0, 2, 1, 3).reshape(B, N, C)

        x_s = self.out_proj(x_s)
        x_c = self.out_proj2(x_c)
        return torch.cat([x_s, x_c], dim=-1)

@HEADS.register_module()
class AYHead(BaseDecodeHead):

    def __init__(self, feature_strides, **kwargs):
        super(AYHead, self).__init__(input_transform='multiple_select', **kwargs)
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = dict(embed_dim=(768))
        embedding_dim = decoder_params['embed_dim']
        pool_ratio = 2
        mlp_ratio = 2
        mutil_channel = [64, 128, 320, 512]
        # self.attn4 = global_meta_block(dim1=c4_in_channels//2, num_heads=8, mlp_ratio=mlp_ratio,
        #                                drop_path=0.1, pool_ratio=pool_ratio)
        #
        # self.attn3 = global_meta_block(dim1=c3_in_channels//2, num_heads=4, mlp_ratio=mlp_ratio, drop_path=0.1,
        #                                pool_ratio=pool_ratio * 2)
        #
        # self.attn2 = global_meta_block(dim1=c2_in_channels//2, num_heads=2, mlp_ratio=mlp_ratio, drop_path=0.1,
        #                                pool_ratio=pool_ratio * 4)
        # self.attn1 = global_meta_block(dim1=c1_in_channels//2, num_heads=1, mlp_ratio=mlp_ratio, drop_path=0.1,
        #                                pool_ratio=pool_ratio * 8)
        self.attn4 = global_meta_block(dim1=896, num_heads=8, mlp_ratio=mlp_ratio,
                                       drop_path=0.1, pool_ratio=pool_ratio)

        self.attn3 = global_meta_block(dim1=448, num_heads=4, mlp_ratio=mlp_ratio, drop_path=0.1,
                                       pool_ratio=pool_ratio * 2)

        self.attn2 = global_meta_block(dim1=224, num_heads=2, mlp_ratio=mlp_ratio, drop_path=0.1,
                                       pool_ratio=pool_ratio * 4)
        self.attn1 = global_meta_block(dim1=112, num_heads=1, mlp_ratio=mlp_ratio, drop_path=0.1,
                                       pool_ratio=pool_ratio * 8)

        self.linear_fuse = ConvModule(
            #in_channels=(c1_in_channels + c2_in_channels + c3_in_channels + c4_in_channels),
            #in_channels=(c2_in_channels + c3_in_channels + c4_in_channels),
            in_channels=(112),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.eucb3 = EUCB(in_channels=896, out_channels=448, kernel_size=3, stride=3 // 2)
        self.eucb2 = EUCB(in_channels=448, out_channels=224, kernel_size=3, stride=3 // 2)
        self.eucb1 = EUCB(in_channels=224, out_channels=112, kernel_size=3, stride=3 // 2)
        
      
        
        

      
        # self.dmc = DMC_fusion(mutil_channel, up_kwargs=up_kwargs)

        self.eff4 = EFF(320)
        self.eff3 = EFF(128)
        self.eff2 = EFF(64)

        
        
        self.fem1 = FEM(c1_in_channels,c1_in_channels)
        self.fem2 = FEM(c2_in_channels, c2_in_channels)
        self.fem3 = FEM(c3_in_channels, c3_in_channels)
        self.fem4 = FEM(c4_in_channels, c4_in_channels)
  
        self.ff3 = FreqFusion(448,448)
        self.ff2 = FreqFusion(224, 224)
        self.ff1 = FreqFusion(112, 112)

        self.bfam3 = BFAM(896,448)
        self.bfam2 = BFAM(448, 224)
        self.bfam1 = BFAM(224, 112)

        self.aifi4 = AIFI(896)
        self.aifi3 = AIFI(448)
        self.aifi2 = AIFI(224)
        self.aifi1 = AIFI(112)
        
        
       
        
       

        self.mfcs4 = MCFS(896,16,16)
        self.mfcs3 = MCFS(448,32,32)
        self.mfcs2 = MCFS(224,64,64)
        self.mfcs1 = MCFS(112,128,128)
        
        self.ema3 = EMA(448)
        self.ema2 = EMA(224)
        self.ema1 = EMA(112)

        # self.eucb3 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(896, 448, kernel_size=3, padding=1),  # 改用3x3卷积
        #     nn.ReLU()
        # )
        # self.eucb2 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(448, 224, kernel_size=3, padding=1),  # 改用3x3卷积
        #     nn.ReLU()
        # )
        # self.eucb1 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(224, 112, kernel_size=3, padding=1),  # 改用3x3卷积
        #     nn.ReLU()
        # )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        # inputs[0] = self.fem1(inputs[0])
        # inputs[1] = self.fem2(inputs[1])
        # inputs[2] = self.fem3(inputs[2])
        # inputs[3] = self.fem4(inputs[3])
        c1  = inputs[0]
        c2 = inputs[1]
        c3 = inputs[2]
        c4 = inputs[3]

        # c1, c2, c3, c4 = x
        # c1, c2, c3, c4 = self.dmc(c1, c2, c3, c4)

        # n, _, h4, w4 = c4.shape
        # _, _, h3, w3 = c3.shape
        # _, _, h2, w2 = c2.shape
        # _, _, h1, w1 = c1.shape
        
        
        
       
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape
        # c4 = self.mla4(c4)

        # 
        # 
        # _c4 = self.ema4(c4)
        # c4 = self.aifi(c4)
        # c4 = c4 + self.mfcs4(c4)

        c4 = self.mfcs4(c4) + c4
        _c4 = c4.flatten(2).transpose(1, 2)

        a4 = self.attn4(_c4, h4, w4)
        
      
        #
        c4 = a4.permute(0, 2, 1).reshape(n, -1, h4, w4)
       
       
        d4 = self.eucb3(c4)

        
        #_c4 = resize(_c4, size=(h2, w2), mode='bilinear', align_corners=False)

        # c3 = c3 + self.mfcs3(c3)
        c3 = self.mfcs3(c3) + c3
        c3 = c3.flatten(2).transpose(1, 2)
        #
        a3 = self.attn3(c3,h3,w3)
        c3 = a3.permute(0, 2, 1).reshape(n, -1, h3, w3)
        # c3 = self.ema3(c3)
        # c3 = self.mla3(c3)
        
        c3 = self.bfam3(c3,d4)
        
        #c3 = c3 + d4
        #c3 = self.ema3(c3)

        


        # _c3 = self.ema3(c3)
        d3 = self.eucb2(c3)
        #_c3 = resize(_c3, size=(h2, w2), mode='bilinear', align_corners=False)
        # c2 = c2 + self.mfcs2(c2)
        c2 = self.mfcs2(c2) + c2
        c2 = c2.flatten(2).transpose(1, 2)
        a2 = self.attn2(c2,h2,w2)
        c2 = a2.permute(0, 2, 1).reshape(n, -1, h2, w2)
        # c2 = self.mla2(c2)
        # c2 = self.ema2(a2)

        c2 = self.bfam2(c2,d3)
        #c2 = c2 + d3
        #c2 = self.ema2(c2)

        # _c2 = self.ema2(c2)
        d2 = self.eucb1(c2)
        #_c2 = resize(_c2, size=(h2, w2), mode='bilinear', align_corners=False)
        
        # c1 = self.meem(c1) + c1
        # c1 = self.ema1(c1)
        # c1 = c1 + self.mfcs1(c1)

        c1 = self.mfcs1(c1) + c1
        c1 = c1.flatten(2).transpose(1, 2)
        c1 = self.attn1(c1,h1,w1)
        c1 = c1.permute(0, 2, 1).reshape(n, -1, h1, w1)
        # c1 = self.mla1(c1)

        c1 = self.bfam1(c1, d2)
        
        # c1 = self.dlk(c1)
        # c1 = self.aifi1(c1)
        #c1 = c1 + d2

        #c1 = self.ema1(c1)
        # c1 = c1 + m1 + d2
     
        #_c1 = self.attn1(c1, h1, w1)

        #_c1 = resize(_c1, size=(h1, w1), mode='bilinear', align_corners=False)
        #_c1 = F.interpolate(_c1, size=(64, 64), mode='bilinear', align_corners=False)

        #_c = self.linear_fuse(torch.cat([_c4, _c3, _c2,_c1], dim=1))
        #_c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))
        
        
        _c = self.linear_fuse(c1)





        x = self.dropout(_c)
        x = self.linear_pred(x)


        return x












