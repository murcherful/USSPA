import os
import sys
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
basic shape:  [bs, feature_channel, point_num]
'''

class MlpConv(nn.Module):
    def __init__(self, input_channel, channels, activation_function=None):
        super(MlpConv, self).__init__()
        self.layer_num = len(channels)
        self.net = nn.Sequential()
        last_channel = input_channel
        for i, channel in enumerate(channels):   
            self.net.add_module('Conv1d_%d' % i, nn.Conv1d(last_channel, channel, kernel_size=1))
            if i != self.layer_num - 1:
                self.net.add_module('ReLU_%d' % i, nn.ReLU())
            last_channel = channel
        if activation_function != None:
            self.net.add_module('af', activation_function)

    def forward(self, x):
        return self.net(x)        


class PcnEncoder(nn.Module):
    def __init__(self, input_channel=3, out_c=1024):
        super(PcnEncoder, self).__init__()
        self.mlp_conv_1 = MlpConv(input_channel, [128, 256])
        self.mlp_conv_2 = MlpConv(512, [512, out_c])

    def forward(self, x):
        '''
        x : [B, 3, N]
        '''
        point_num = x.shape[2]
        x = self.mlp_conv_1(x)

        x_max = torch.max(x, 2).values
        x_max = torch.unsqueeze(x_max, 2)
        x_max = x_max.repeat(1, 1, point_num) 
        x = torch.cat([x, x_max], 1)
        
        x = self.mlp_conv_2(x)
        
        x_max = torch.max(x, 2).values
        return x_max
    
class PcnEncoder2(nn.Module):
    def __init__(self, input_channel=3, out_c=1024):
        super().__init__()
        self.mlp_conv_1 = MlpConv(input_channel, [128, 256])
        self.mlp_conv_2 = MlpConv(512, [512, out_c])

    def forward(self, x):
        '''
        x : [B, N, 3]
        '''
        B, N, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.mlp_conv_1(x)

        x_max = torch.max(x, 2, keepdim=True).values
        x_max = x_max.repeat(1, 1, N) 
        x = torch.cat([x, x_max], 1)
        
        x = self.mlp_conv_2(x)
        
        x_max = torch.max(x, 2, keepdim=True).values
        return x_max


class PcnDecoder(nn.Module): 
    def __init__(self):
        super(PcnDecoder, self).__init__()
        self.num_coarse = 1024
        self.grid_scale = 0.05
        self.grid_size = 4
        self.num_fine = (self.grid_size**2) * self.num_coarse
        #self.mlp_conv_1 = MlpConv(1024, [1024, 1024, self.num_coarse*3])

        coarse_lst = [1024, 1024, self.num_coarse*3]
        in_features = 1024
        decoder_lst = []
        for i in range(len(coarse_lst)):
            decoder_lst.append(nn.Linear(in_features, coarse_lst[i]))
            in_features = coarse_lst[i]
        self.mlp_1 = nn.Sequential(*decoder_lst)

        self.mlp_conv_2 = MlpConv(1024+3+2, [512, 512, 3])

    def forward(self, x):
        ### Decoder coarse
        fd1 = self.mlp_1(x)
        coarse = fd1.view(-1, self.num_coarse, 3)

        ### Folding
        g1 = torch.linspace(self.grid_scale*(-1), self.grid_scale, self.grid_size).cuda()
        g2 = torch.linspace(self.grid_scale*(-1), self.grid_scale, self.grid_size).cuda()
        grid = torch.meshgrid(g1, g2)
        grid = torch.reshape(torch.stack(grid, dim=2), (1, -1, 2))
        grid_feat = grid.repeat([x.shape[0], self.num_coarse, 1])
        point_feat = coarse.unsqueeze(2).repeat([1, 1, self.grid_size**2, 1])
        point_feat = torch.reshape(point_feat, (-1, self.num_fine, 3))
        glob_feat = x.unsqueeze(1).repeat([1, self.num_fine, 1])

        feat = torch.cat([grid_feat, point_feat, glob_feat], dim=2)
        fine = self.mlp_conv_2(feat.permute(0, 2, 1))
        fine = fine.permute(0, 2, 1) + point_feat

        return coarse, fine

class PcnDecoder2(nn.Module): 
    def __init__(self, grid_size=4, has_coarse=False, num_coarse=None):
        super().__init__()
        self.grid_scale = 0.05
        self.grid_size = grid_size
        if has_coarse:
            self.num_coarse = num_coarse
        else:
            self.num_coarse = 1024
        self.num_fine = (self.grid_size**2) * self.num_coarse
        #self.mlp_conv_1 = MlpConv(1024, [1024, 1024, self.num_coarse*3])
        self.has_coarse = has_coarse
        if not self.has_coarse:
            coarse_lst = [1024, 1024, self.num_coarse*3]
            in_features = 1024
            decoder_lst = []
            for i in range(len(coarse_lst)):
                decoder_lst.append(nn.Linear(in_features, coarse_lst[i]))
                in_features = coarse_lst[i]
            self.mlp_1 = nn.Sequential(*decoder_lst)

        self.mlp_conv_2 = MlpConv(1024+3+2, [512, 512, 3])

    def forward(self, x, coarse=None):
        ### Decoder coarse
        
        if not self.has_coarse: 
            fd1 = self.mlp_1(x)
            coarse = fd1.view(-1, self.num_coarse, 3)

        ### Folding
        g1 = torch.linspace(self.grid_scale*(-1), self.grid_scale, self.grid_size).cuda()
        g2 = torch.linspace(self.grid_scale*(-1), self.grid_scale, self.grid_size).cuda()
        grid = torch.meshgrid(g1, g2)
        grid = torch.reshape(torch.stack(grid, dim=2), (1, -1, 2))
        grid_feat = grid.repeat([x.shape[0], self.num_coarse, 1])
        point_feat = coarse.unsqueeze(2).repeat([1, 1, self.grid_size**2, 1])
        point_feat = torch.reshape(point_feat, (-1, self.num_fine, 3))
        glob_feat = x.unsqueeze(1).repeat([1, self.num_fine, 1])

        feat = torch.cat([grid_feat, point_feat, glob_feat], dim=2)
        fine = self.mlp_conv_2(feat.permute(0, 2, 1))
        fine = fine.permute(0, 2, 1) + point_feat

        return coarse, fine

class TopNetNode(nn.Module):
    def __init__(self, input_channel, append_channel, output_channel, output_num, activation_function=None):
        super(TopNetNode, self).__init__()
        self.append_channel = append_channel
        self.output_channel = output_channel 
        self.output_num = output_num
        self.mlp_conv = MlpConv(input_channel+append_channel, [512, 256, 64, output_channel*output_num], activation_function=activation_function)
    
    '''
    append_x shape: [bs, feature_channel, 1]
    '''
    def forward(self, x, append_x=None):
        batch_size = x.shape[0]
        point_num = x.shape[2]
        if self.append_channel != 0:
            append_x = append_x.repeat(1, 1, point_num)
            x = torch.cat([x, append_x], 1)
        x = self.mlp_conv(x)
        x = torch.reshape(x, (batch_size, self.output_channel, -1))
        return x


class TopNetDecoder(nn.Module):
    def __init__(self, input_channel, output_nums, get_all_res=False):
        super(TopNetDecoder, self).__init__()
        self.get_all_res = get_all_res
        self.topnet_node_0 = TopNetNode(input_channel, 0, 8, output_nums[0], activation_function=nn.Tanh())
        self.topnet_nodes = []
        
        for output_num in output_nums[1:-1]:
            self.topnet_nodes.append(TopNetNode(8, input_channel, 8, output_num))
        self.topnet_nodes.append(TopNetNode(8, input_channel, 3, output_nums[-1]))
        self.topnet_nodes = nn.ModuleList(self.topnet_nodes)

    '''
    x shape: [bs, feature_channel, 1] or [bs, feature_channel]
    '''
    def forward(self, x):
        node_res = []
        if len(x.shape) == 2:
            global_x = torch.unsqueeze(x, 2)
        else:
            assert(len(x.shape) == 3 and x.shape[2] == 1)
            global_x = x
        res = self.topnet_node_0(global_x)
        node_res.append(res)
        for topnet_node in self.topnet_nodes:
            res = topnet_node(res, global_x)
            node_res.append(res)
        res = torch.permute(res, [0, 2, 1])
        if self.get_all_res:
            return res, node_res
        else:
            return res


def get_k_neighbor(points, k=8):
    '''
    points: [B, N, 3]   Tensor
    k:                  int
    '''
    x = points
    x1 = x.unsqueeze(1)
    x2 = x.unsqueeze(2)
    diff = (x1-x2).norm(dim=-1)
    _, idx = diff.topk(k, largest=False)
    return idx.int()

class UpSampleModule(nn.Module):
    def __init__(self, times=4, gf_c=128):
        super().__init__()
        BC = 128
        self.mlp1 = MlpConv(3+gf_c, [BC, BC])
        self.mlp2 = MlpConv(3+BC*2, [BC, BC])
        self.mlp3 = MlpConv(3+BC*2, [BC, 3*times])
        self.times = times 
    
    def forward(self, p, gf):
        '''
        p  : [B, N, 3]
        gf : [B, gf_c, 1]

        out: [B, N*times, 3]
        '''
        p = p.permute(0, 2, 1)
        B, _, N = p.shape
        gf = gf.repeat([1, 1, N])
        x = torch.cat([p, gf], 1)
        x = self.mlp1(x)
        x_max = torch.max(x, dim=2, keepdim=True).values.repeat([1, 1, N])
        x = torch.cat([p, x, x_max], 1)
        x = self.mlp2(x)
        x_max = torch.max(x, dim=2, keepdim=True).values.repeat([1, 1, N])
        x = torch.cat([p, x, x_max], 1)
        x = self.mlp3(x) 
        p = p.repeat([1, self.times, 1])
        p += x 
        p = p.permute(0, 2, 1).reshape([B, N*self.times, 3])
        return p


if __name__ == '__main__1':
    pass
    '''
    mlp_conv = MlpConv(3, [128, 256], nn.ReLU())
    '''
    
    point_num = 10
    device = torch.device('cuda')
    model = PcnEncoder().to(device)
    x = torch.randn(2, point_num, 3).to(device)
    x = x.permute(0, 2, 1)
    x = model(x)
    #model2 = TopNetDecoder(1024, [4, 8, 8, 8])
    model2 = PcnDecoder().to(device)
    x = model2(x)
    print(x[0].shape, x[1].shape)
    print(model)
    print(model2)

if __name__ == '__main__':
    points = torch.randn(4, 2048, 3)
    idx = get_k_neighbor(points)
    print(idx.shape)
