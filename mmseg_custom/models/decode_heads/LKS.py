import torch
from torch import nn
from timm.models.layers import DropPath
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
# from .MEEM import EdgeEnhancer
from .FADC import FrequencySelection
from .APCconv import PConv
# from .FourierUnit import FourierUnit
from .DSConv_pro import DSConv_pro
from .EMCAD import SAB
from .FEM import FEM
from .WTConv import WTConv2d
import numpy as np
from .TVConv import TVConv
from .LWN import LWN
from .TripletAttention import TripletAttention,ChannelPool
from .odconv import ODConv2d
from .CBAM import CBAM
from .FADC import AdaptiveDilatedConv


class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, down_kernel=5, down_stride=4):
        super().__init__()

        
        self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")  
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")  
        self.act = nn.GELU()  
       
        
        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2,
                              groups=in_dim)
        self.proj = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, stride=1, padding=0)  

        self.apply(self._init_weights)  

    def _init_weights(self, m):
       
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)  
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
       
        dx = self.down(x)  
        
        udx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False) 
        
        lx = self.norm1(self.lconv(self.act(x * udx)))  
        hx = self.norm2(self.hconv(self.act(x - udx)))  

        out = self.act(self.proj(torch.cat([lx, hx], dim=1)))  

        return out

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 5, 1, (5 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 5, 1, (5 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x
 
class DLK_2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
      
        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)

       
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)

        att = torch.cat([att1, att2], dim=1)
       
        avg_att = torch.mean(att, dim=1, keepdim=True)
      
        max_att, _ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:, 0, :, :].unsqueeze(1) + att2 * att[:, 1, :, :].unsqueeze(1)
        output = output + x
        return output




class MCFS(nn.Module):
    def __init__(self, dim,H,W, s_kernel_size=3):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1, padding=0)
        self.proj_2 = nn.Conv2d(dim * 2, dim, 1, padding=0)
        self.proj_3 = nn.Conv2d(dim * 3, dim, 1, padding=0)
        self.norm_proj = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # multiscale spatial context layers
        self.s_ctx_1 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, padding=s_kernel_size // 2, groups=dim // 4)
        self.s_ctx_2 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, dilation=2, padding=(s_kernel_size // 2) * 2,
                                 groups=dim // 4)
        # self.s_ctx_1 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, padding=s_kernel_size // 2, groups=dim // 4)
        # self.s_ctx_2 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, dilation=3, padding=(s_kernel_size // 2) * 3,
        #                          groups=dim // 4)
        # self.s_ctx_3 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, dilation=5, padding=(s_kernel_size // 2) * 5,
        #                          groups=dim // 4)
        # self.s_ctx_4 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, dilation=7, padding=(s_kernel_size // 2) * 7,
        #                          groups=dim // 4)
        # self.s_ctx_5 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, dilation=9, padding=(s_kernel_size // 2) * 9,
        #                          groups=dim // 4)

        self.norm_s = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # sharpening module layers
        # self.h_ctx = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False, groups=dim)
        self.h_ctx = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False, groups=dim)
        self.h_ctx1 = nn.Conv2d(dim, dim, kernel_size=5, padding=4, dilation=2,bias=False, groups=dim)
     
        self.norm_h = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.act = nn.GELU()
        self.odcon = ODConv2d(dim,dim, 3)
        # self.edge = EdgeEnhancer(dim, norm=nn.BatchNorm2d, act=nn.ReLU)
        # self.adp = AdaptiveDilatedConv(dim, dim, 3)
        # self.pcov = PConv(dim, dim, k=3, s=1)
        # 
        #self.snake_conv = DSConv_pro(in_channels=dim, out_channels=dim, kernel_size=9,
        #                              extend_scope=1.0, morph=0, if_offset=True)
        # self.snake_convy = DSConv_pro(in_channels=dim, out_channels=dim, kernel_size=9,
        #                               extend_scope=1.0, morph=1, if_offset=True)
        # self.conv1 = EncoderConv(2 * dim, dim)
        
        # self.DoGSharpening = DoGSharpening(dim)
        # self.simam = simam_module()
        # self.laplace_sharpening = LaplaceSharpening(dim)
        # self.wtconv = WTConv2d(dim,dim)
        # self.conv_spatial = nn.Conv2d(
        #     dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # self.conv_spatial = nn.Conv2d(
        #     dim, dim, 7, stride=1, padding=3, groups=dim)
        # self.sab = SAB()
        # self.fem = FEM(dim,dim)
        # self.lka = MLKA_Ablation(dim)
        # self.fs = FrequencySelection(dim)
        #self.t = TripletAttention(dim)
        self.n =nn.Sequential(nn.Conv2d(dim, dim * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim * 4),
                nn.Conv2d(dim * 4, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim))
        #self.block = Block(dim)
        #self.compress = ChannelPool()
        #self.si = simam_module()
        #self.cbam = CBAM(dim,dim)
       
        self.dlk = DLK_2D(dim)
        
        
    def forward(self, x):
      
        x = self.norm_proj(self.act(self.proj_1(x)))
        
        B, C, H, W = x.shape

        # extract multi-scale contextual features


        
        
        #sx1 = self.act(self.s_ctx_1(x))
        #sx1 = self.act(self.s_ctx_1(x))



        #sx2 = self.act(self.s_ctx_2(x))
        
       

        #sx = self.norm_s( sx1 + sx2)
        sx = self.dlk(x)
        
       
        
        
       

        
       
        # feature enhancement using learnable sharpening factors
        # implementation of sharpening module
       
      
        hx = self.act(self.h_ctx(x))
       
       
        

        

         

        # hx = self.act(self.conv_spatial(x))

        # hx_t = x - hx.mean(dim=1).unsqueeze(1)

        
        # hx_t = self.sab(hx_t) * hx_t
        
       
     
        hx_t = x - hx.mean(dim=1).unsqueeze(1)
        hx_t = torch.softmax(hx.mean(dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1), dim=1) * hx_t

        hx = self.norm_h(hx + hx_t)
        
        

        # combine the multiscale contetxual features with the sharpened features
        x = self.act(self.proj_2(torch.cat([sx, hx], dim=1)))
       
        

        return x


class FSB(nn.Module):
    """
    Feature Sharpening Block:
    It is the core block of the COSNet encoder/backbone,
    utilized to extract semantically rich features for segementation task in cluttered background.
    -----------------------------------------------
    dim:           Input channel dimension
    s_kernel_size: Kernel size for spatial context layers
    expan_ratio:   Expansion ratio used for channels in MLP
    ------------------------------------------------

    """

    def __init__(self, dim, s_kernel_size=3, drop_path=0.1, layer_scale_init_value=1e-6, expan_ratio=4):
        super().__init__()

        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm_dw = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.layer_norm_1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm_2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.mlp = MLP(dim=dim, mlp_ratio=expan_ratio)
        self.attn = MCFS(dim, s_kernel_size=s_kernel_size)

        self.drop_path_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = x + self.norm_dw(self.act(self.conv_dw(x)))

        x_copy = x
        x = self.layer_norm_1(x_copy)
        x = self.drop_path_1(self.attn(x))
        out = x + x_copy

        x = self.layer_norm_2(out)
        x = self.drop_path_2(self.mlp(x))
        out = out + x

        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.fc_1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc_2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.fc_1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc_2(x)

        return x


class BEM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        dx = self.pool(x)
        ex = torch.nn.functional.interpolate(dx, size=x.shape[2:], mode='bilinear') - x
        x = torch.cat([ex, x], dim=1)
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)

        return x
