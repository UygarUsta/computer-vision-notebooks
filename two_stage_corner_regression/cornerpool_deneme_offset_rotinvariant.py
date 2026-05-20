from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import collections
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from corner_modules_v2 import RotInvCornerHead, PatchRefinement

# <<<<<<<<< START OF CORNER POOLING IMPLEMENTATION >>>>>>>>>

class TopPool(nn.Module):
    def forward(self, x):
        return torch.cummax(x.flip(dims=(2,)), dim=2)[0].flip(dims=(2,))

class BottomPool(nn.Module):
    def forward(self, x):
        return torch.cummax(x, dim=2)[0]

class LeftPool(nn.Module):
    def forward(self, x):
        return torch.cummax(x.flip(dims=(3,)), dim=3)[0].flip(dims=(3,))

class RightPool(nn.Module):
    def forward(self, x):
        return torch.cummax(x, dim=3)[0]

class CornerPoolModule(nn.Module):
    def __init__(self, in_channels, corner_type):
        super().__init__()
        assert corner_type in ['tl', 'tr', 'br', 'bl']
        self.corner_type = corner_type

        if corner_type == 'tl':
            self.pool1 = TopPool()
            self.pool2 = LeftPool()
        elif corner_type == 'tr':
            self.pool1 = TopPool()
            self.pool2 = RightPool()
        elif corner_type == 'bl':
            self.pool1 = BottomPool()
            self.pool2 = LeftPool()
        elif corner_type == 'br':
            self.pool1 = BottomPool()
            self.pool2 = RightPool()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        x = p1 + p2
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class CornerHead(nn.Module):
    def __init__(self, in_channels, out_channels, corner_type):
        super().__init__()
        inter_channels = 256
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.corner_pool = CornerPool(corner_type=corner_type)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.pred_conv = nn.Conv2d(inter_channels, out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = self.conv_block1(x)
        pooled = self.corner_pool(shortcut)
        pooled_processed = self.conv_block2(pooled)
        out = shortcut + pooled_processed
        out = self.pred_conv(out)
        return out

# <<<<<<<<< END OF CORNER POOLING IMPLEMENTATION >>>>>>>>>

# <<<<<<<<< YENI: COARSE-TO-FINE REFINEMENT MODULU >>>>>>>>>
class CoarseToFineRefinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 4 köşeden (tl, tr, br, bl) gelen feature'lar concat edileceği için in_channels * 4
        self.conv1 = nn.Conv2d(in_channels * 4, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 8, kernel_size=1)
        
        init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x, wh_coarse):
        B, C, H, W = x.shape
        
        # Grid oluşturma
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        center_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).float() # [1, 2, H, W]

        fine_features = []
        wh_coarse_detached = wh_coarse.detach()

        
        for i in range(4): # 4 köşe için dön
            # Coarse ofseti al (dx, dy)
            #offset = wh_coarse[:, i*2:(i+1)*2, :, :] 
            offset = wh_coarse_detached[:, i*2:(i+1)*2, :, :] 
            
            # Feature map üzerindeki mutlak konumu bul
            pos = center_grid + offset 
            
            # Koordinatları grid_sample'ın istediği [-1, 1] aralığına normalize et
            pos_x = (pos[:, 0, :, :] / (W - 1)) * 2 - 1
            pos_y = (pos[:, 1, :, :] / (H - 1)) * 2 - 1
            
            sample_grid = torch.stack([pos_x, pos_y], dim=-1) # [B, H, W, 2]
            
            # Köşenin olduğu bölgeden bilinear interpolation ile özellik çek
            sampled_feat = F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            fine_features.append(sampled_feat)
            
        # 4 köşeden çekilen özellikleri birleştir [B, C*4, H, W]
        concat_features = torch.cat(fine_features, dim=1) 
        
        # Ince ofset (delta_wh) tahmini yap
        out = self.conv1(concat_features)
        out = self.relu(out)
        delta_wh = self.conv2(out)
        
        # Kaba tahminin üzerine ince delta ofsetini ekleyerek nihai sonucu dön
        return wh_coarse + delta_wh
# <<<<<<<<< -------------------------------------- >>>>>>>>>

# <<<<<<<<< YENI: COARSE-TO-FINE REFINEMENT >>>>>>>>>
# class CoarseToFineRefinement(nn.Module):
#     def __init__(self, in_channels, patch_size=3):
#         super().__init__()
#         self.patch_size = patch_size
        
#         # 1. KANAL SIKIŞTIRMA (VRAM patlamasını önlemek için)
#         # 256 kanallı giriş özelliklerini 32 kanala düşürüyoruz.
#         self.reduce_channels = 32
#         self.reduce_conv = nn.Conv2d(in_channels, self.reduce_channels, kernel_size=1)
        
#         # 2. CONCAT EDİLECEK KANALLARIN HESABI
#         # Merkez özellik (32) + [4 köşe * (3x3 = 9 nokta) * 32 kanal] = 32 + 1152 = 1184 Kanal
#         num_points_per_corner = patch_size ** 2
#         concat_channels = self.reduce_channels + (4 * num_points_per_corner * self.reduce_channels)
        
#         # 3. İŞLEYİCİ KATMANLAR
#         self.conv1 = nn.Conv2d(concat_channels, 256, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(256, 8, kernel_size=1)
        
#         # Ağırlıkların başlatılması
#         init.kaiming_uniform_(self.reduce_conv.weight, nonlinearity='relu')
#         nn.init.constant_(self.reduce_conv.bias, 0)
#         init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
#         nn.init.constant_(self.conv1.bias, 0)
#         init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
#         nn.init.constant_(self.conv2.bias, 0)

#     def forward(self, x, wh_coarse):
#         B, C, H, W = x.shape
        
#         # Adım 1: Özellik haritasını daralt [B, 32, H, W]
#         reduced_x = self.relu(self.reduce_conv(x))
        
#         # Grid oluşturma
#         grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
#         center_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).float() # [1, 2, H, W]

#         fine_features = []
        
#         # Global bağlamı (merkezin kendi dokusunu) listeye en başta ekle
#         fine_features.append(reduced_x)
        
#         # Kaba tahminleri gradyandan kopar (Kaba tahmini sadece lokasyon bulmak için kullan!)
#         wh_coarse_detached = wh_coarse.detach()
        
#         # 3x3 patch için merkezin etrafında dolaşacağımız yarıçap (1)
#         radius = self.patch_size // 2 

#         for i in range(4): # 4 köşe için dön
#             # Coarse ofseti al (dx, dy)
#             offset = wh_coarse_detached[:, i*2:(i+1)*2, :, :] 
            
#             # Kaba köşenin olduğu mutlak merkez konum
#             base_pos = center_grid + offset 
            
#             # MANUEL PATCH SAMPLING: -1, 0, 1 piksellik kaydırmalarla 9 nokta çek
#             for dy in range(-radius, radius + 1):
#                 for dx in range(-radius, radius + 1):
                    
#                     # Konuma dy ve dx ekleyerek yama (patch) üzerindeki o anki noktayı bul
#                     pos_x = base_pos[:, 0, :, :] + dx
#                     pos_y = base_pos[:, 1, :, :] + dy
                    
#                     # [-1, 1] aralığına normalize et
#                     norm_pos_x = (pos_x / (W - 1)) * 2 - 1
#                     norm_pos_y = (pos_y / (H - 1)) * 2 - 1
                    
#                     sample_grid = torch.stack([norm_pos_x, norm_pos_y], dim=-1) # [B, H, W, 2]
                    
#                     # O tek noktadan özelliği çek (padding_mode='border' hayati önem taşır)
#                     sampled_feat = F.grid_sample(reduced_x, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
#                     fine_features.append(sampled_feat)
            
#         # 1 merkez + 36 yama noktası birleştiriliyor [B, 1184, H, W]
#         concat_features = torch.cat(fine_features, dim=1) 
        
#         # Ince ofset (delta_wh) tahmini yap
#         out = self.conv1(concat_features)
#         out = self.relu(out)
#         delta_wh = self.conv2(out)
        
#         # Gradyanı akan orijinal wh_coarse üzerine ince delta ofsetini ekle
#         return wh_coarse + delta_wh
# # <<<<<<<<< ----------------------------------------------------- >>>>>>>>>


class CornerHeadRotationInvariant(nn.Module):
    """
    Döndürülmüş (Rotated) belgelerde CornerPool'un yönelimsel kısıtlamalarını kaldıran,
    dokusal (textural) özelliklerle köşe tespit eden modül.
    """
    def __init__(self, in_channels):
        super().__init__()
        inter_channels = 256
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.conv2(x)
        return out

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)

class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1',ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2',DWConvLayer(out_channels, out_channels, stride=stride))
    def forward(self, x):
        return super().forward(x)

class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias=False):
        super().__init__()
        groups = in_channels
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups, momentum=BN_MOMENTUM))
    def forward(self, x):
        return super().forward(x)

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.out_channels = out_channels
        pad = kernel//2 if padding == 0 else padding
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
        self.add_module('relu', nn.ReLU(True))
    def forward(self, x):
        return super().forward(x)

class BRLayer(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
    def forward(self, x):
        return super().forward(x)

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0

        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t-1) or (i%2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

class HarDBlock_v2(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.insert(0, k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def __init__(self, in_channels, growth_rate, grmul, n_layers, dwconv=False):
        super().__init__()
        self.links = []
        conv_layers_ = []
        bnrelu_layers_ = []
        self.layer_bias = []
        self.out_channels = 0
        self.out_partition = collections.defaultdict(list)

        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            for j in link:
                self.out_partition[j].append(outch)

        cur_ch = in_channels
        for i in range(n_layers):
            accum_out_ch = sum( self.out_partition[i] )
            real_out_ch = self.out_partition[i][0]
            conv_layers_.append( nn.Conv2d(cur_ch, accum_out_ch, kernel_size=3, stride=1, padding=1, bias=True) )
            bnrelu_layers_.append( BRLayer(real_out_ch) )
            cur_ch = real_out_ch
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += real_out_ch
        self.conv_layers = nn.ModuleList(conv_layers_)
        self.bnrelu_layers = nn.ModuleList(bnrelu_layers_)

    def transform(self, blk, trt=False):
        in_ch = blk.layers[0][0].weight.shape[1]
        for i in range(len(self.conv_layers)):
            link = self.links[i].copy()
            link_ch = [blk.layers[k-1][0].weight.shape[0] if k > 0 else blk.layers[0][0].weight.shape[1] for k in link]
            part = self.out_partition[i]
            w_src = blk.layers[i][0].weight
            b_src = blk.layers[i][0].bias

            self.conv_layers[i].weight[0:part[0], :, :,:] = w_src[:, 0:in_ch, :,:]
            self.layer_bias.append(b_src)
            if b_src is not None:
                if trt:
                    self.conv_layers[i].bias[1:part[0]] = b_src[1:]
                    self.conv_layers[i].bias[0] = b_src[0]
                    self.conv_layers[i].bias[part[0]:] = 0
                    self.layer_bias[i] = None
                else:
                    self.conv_layers[i].bias = None
            else:
                self.conv_layers[i].bias = None

            in_ch = part[0]
            link_ch.reverse()
            link.reverse()
            if len(link) > 1:
                for j in range(1, len(link) ):
                    ly  = link[j]
                    part_id  = self.out_partition[ly].index(part[0])
                    chos = sum( self.out_partition[ly][0:part_id] )
                    choe = chos + part[0]
                    chis = sum( link_ch[0:j] )
                    chie = chis + link_ch[j]
                    self.conv_layers[ly].weight[chos:choe, :,:,:] = w_src[:, chis:chie,:,:]

            self.bnrelu_layers[i] = None
            if isinstance(blk.layers[i][1], nn.BatchNorm2d):
                self.bnrelu_layers[i] = nn.Sequential(blk.layers[i][1], blk.layers[i][2])
            else:
                self.bnrelu_layers[i] = blk.layers[i][1]

    def forward(self, x):
        layers_ = []
        outs_ = []
        xin = x
        for i in range(len(self.conv_layers)):
            link = self.links[i]
            part = self.out_partition[i]

            xout = self.conv_layers[i](xin)
            layers_.append(xout)

            xin = xout[:,0:part[0],:,:] if len(part) > 1 else xout
            if self.layer_bias[i] is not None:
                xin += self.layer_bias[i].view(1,-1,1,1)

            if len(link) > 1:
                for j in range( len(link) - 1 ):
                    ly  = link[j]
                    part_id  = self.out_partition[ly].index(part[0])
                    chs = sum( self.out_partition[ly][0:part_id] )
                    che = chs + part[0]
                    xin += layers_[ly][:,chs:che,:,:]

            xin = self.bnrelu_layers[i](xin)

            if i%2 == 0 or i == len(self.conv_layers)-1:
                outs_.append(xin)

        out = torch.cat(outs_, 1)
        return out

class HarDNetBase(nn.Module):
    def __init__(self, arch, depth_wise=False):
        super().__init__()
        if arch == 85:
            first_ch  = [48, 96]
            second_kernel = 3
            ch_list = [  192, 256, 320, 480, 720]
            grmul = 1.7
            gr      = [   24,  24,  28,  36,  48]
            n_layers = [   8,  16,  16,  16,  16]
        elif arch == 68:
            first_ch  = [32, 64]
            second_kernel = 3
            ch_list = [  128, 256, 320, 640]
            grmul = 1.7
            gr      = [   14,  16,  20,  40]
            n_layers = [   8,  16,  16,  16]
        else:
            exit()

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        self.base.append (ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2,  bias=False))
        self.base.append (ConvLayer(first_ch[0], first_ch[1],  kernel=second_kernel))
        self.base.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i != blks-1:
                self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if i== 0:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))
            elif i != blks-1 and i != 1 and i != 3:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_uniform_(m.state_dict()[key], nonlinearity='relu')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
    def forward(self, x, skip, concat=True):
        out = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode="bilinear", align_corners=True)
        if concat:
            out = torch.cat([out, skip], 1)
        return out

class HarDNetSeg(nn.Module):
    def __init__(self, num_layers, heads, pretrained, down_ratio, final_kernel, last_level, head_conv, out_channel=0, trt=False):
        super(HarDNetSeg, self).__init__()
        self.trt = trt
        self.first_level = int(np.log2(down_ratio))-1
        self.last_level = last_level

        self.base = HarDNetBase(num_layers).base
        self.last_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        if num_layers == 85:
            self.last_proj = ConvLayer(784, 256, kernel=1)
            self.last_blk = HarDBlock(768, 80, 1.7, 8)
            self.skip_nodes = [1,3,8,13]
            self.SC = [32, 32, 0]
            gr = [64, 48, 28]
            layers = [8, 8, 4]
            ch_list2 = [224 + self.SC[0], 160 + self.SC[1], 96 + self.SC[2]]
            channels = [96, 214, 458, 784]
            self.skip_lv = 3
            if pretrained:
                try:
                    weights = torch.load('/opt/projeler/detection_codes/modelSave/hardnet85_base.pth')
                    self.base.load_state_dict(weights)
                except FileNotFoundError:
                    pass

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up      = nn.ModuleList([])
        self.avg9x9  = nn.AvgPool2d(kernel_size=(9,9), stride=1, padding=(4,4))
        prev_ch = self.last_blk.get_out_ch()

        for i in range(3):
            skip_ch = channels[3-i]
            self.transUpBlocks.append(TransitionUp(prev_ch, prev_ch))
            if i < self.skip_lv:
                cur_ch = prev_ch + skip_ch
            else:
                cur_ch = prev_ch
            self.conv1x1_up.append(ConvLayer(cur_ch, ch_list2[i], kernel=1))
            cur_ch = ch_list2[i]
            cur_ch -= self.SC[i]
            cur_ch *= 3
            blk = HarDBlock(cur_ch, gr[i], 1.7, layers[i])
            self.denseBlocksUp.append(blk)
            prev_ch = blk.get_out_ch()

        prev_ch += self.SC[0] + self.SC[1] + self.SC[2]

        weights_init(self.denseBlocksUp)
        weights_init(self.conv1x1_up)
        weights_init(self.last_blk)
        weights_init(self.last_proj)
        
        self.heads = heads
        self.corner_heads = nn.ModuleDict()
        
        # İNCE AYAR MODÜLÜ (Coarse to Fine) BURAYA EKLENIYOR
        #self.wh_refinement = CoarseToFineRefinement(prev_ch)
        self.wh_refinement = PatchRefinement(prev_ch)

        for head in self.heads:
            classes = self.heads[head]
            if head == 'corners':
                self.corner_heads['tl'] = RotInvCornerHead(prev_ch)
                self.corner_heads['tr'] = RotInvCornerHead(prev_ch)
                self.corner_heads['br'] = RotInvCornerHead(prev_ch)
                self.corner_heads['bl'] = RotInvCornerHead(prev_ch)
                fill_fc_weights(self.corner_heads)
                # self.corner_heads['tl'].conv2.bias.data.fill_(-4.59)
                # self.corner_heads['tr'].conv2.bias.data.fill_(-4.59)
                # self.corner_heads['br'].conv2.bias.data.fill_(-4.59)
                # self.corner_heads['bl'].conv2.bias.data.fill_(-4.59)
                continue
            
            if head_conv > 0:
                ch = max(128, classes*4)
                fc = nn.Sequential(
                    nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                fill_fc_weights(fc)
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                fill_fc_weights(fc)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
            self.__setattr__(head, fc)

    def forward(self, x):
        xs = []
        x_sc = []

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.skip_nodes:
                xs.append(x)

        x = self.last_proj(x)
        x = self.last_pool(x)
        x2 = self.avg9x9(x)
        x3 = x/(x.sum((2,3),keepdim=True) + 0.1)
        x = torch.cat([x,x2,x3],1)
        x = self.last_blk(x)

        for i in range(3):
            skip_x = xs[3-i]
            x = self.transUpBlocks[i](x, skip_x, (i<self.skip_lv))
            x = self.conv1x1_up[i](x)
            if self.SC[i] > 0:
                end = x.shape[1]
                x_sc.append( x[:,end-self.SC[i]:,:,:].contiguous() )
                x = x[:,:end-self.SC[i],:,:].contiguous()
            x2 = self.avg9x9(x)
            x3 = x/(x.sum((2,3),keepdim=True) + 0.1)
            x = torch.cat([x,x2,x3],1)
            x = self.denseBlocksUp[i](x)

        scs = [x]
        for i in range(3):
            if self.SC[i] > 0:
                scs.insert(0,  F.interpolate(x_sc[i], size=(x.size(2), x.size(3)), mode="bilinear", align_corners=True) )
        x = torch.cat(scs,1)
        z = {}

        for head in self.heads:
            if head == 'corners':
                tl = self.corner_heads['tl'](x)
                tr = self.corner_heads['tr'](x)
                br = self.corner_heads['br'](x)
                bl = self.corner_heads['bl'](x)
                z[head] = torch.cat([tl, tr, br, bl], dim=1)
            elif head == 'wh':
                wh_coarse = self.__getattr__(head)(x)
                z['wh_coarse'] = wh_coarse
                # Kaba tahmini Refinement modülüne gönderip Ince tahmini üretiyoruz
                z['wh_fine'] = self.wh_refinement(x, wh_coarse) 
            else:
                z[head] = self.__getattr__(head)(x)

        if self.trt:
            return [z[h] for h in sorted(z.keys())]
        
        # Artık 5 değil 6 parametre dönüyoruz (wh ikiye ayrıldı)
        return z["hm"], z["offset"], z["wh_coarse"], z["wh_fine"], z["corners"], z["corner_offset"]

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4, trt=False):
    model = HarDNetSeg(
                 num_layers,
                 heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=4,
                 head_conv=head_conv,
                 trt = trt)
    return model

if __name__ == "__main__":
    model = get_pose_net(85,{"hm":7,"offset":2,"wh":8,"corners":4,"corner_offset":8})
    random_tensor = torch.rand((1,3,512,512))
    # 6 Çıktı karşılanıyor
    out_hm, out_off, out_whc, out_whf, out_cor, out_coroff = model(random_tensor)
    print("HM shape:", out_hm.shape)
    print("WH_Coarse shape:", out_whc.shape)
    print("WH_Fine shape:", out_whf.shape)