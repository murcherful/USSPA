import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch.nn as nn 
import torch 
import torch.nn.functional as F
from pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule, PointnetFPModule, FPS, FPS2, group, three_nn, three_interpolate, PointnetSAModule_test
from base_model_util import MlpConv


class basic_conv1d_seq(nn.Module):
    def __init__(self, channels, BNDP=True):
        super(basic_conv1d_seq, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(channels)-2):
            self.net.add_module('conv1d_%d' % i, nn.Conv1d(channels[i], channels[i+1], 1))
            if BNDP:
                self.net.add_module('bn1d_%d' % i, nn.BatchNorm1d(channels[i+1]))
            self.net.add_module('relu_%d' % i, nn.ReLU())
            if BNDP:
                self.net.add_module('drop_%d' % i, nn.Dropout(0.5))
        self.net.add_module('conv1d_%d' % (len(channels)-2), nn.Conv1d(channels[-2], channels[-1], 1))
    
    def forward(self, x):
        return self.net(x)


class pointnet2_seg(nn.Module):
    def __init__(self, input_channel_num=3):
        '''
        input_channel_num: 3+C
        '''
        super(pointnet2_seg, self).__init__()
        c_in = input_channel_num
        self.sa1 = PointnetSAModuleMSG(512, [0.1, 0.2, 0.4], [32, 64, 128], [[c_in, 32, 32, 64], [c_in, 64, 64, 128], [c_in, 64, 96, 128]])
        c_in = 128+128+64
        self.sa2 = PointnetSAModuleMSG(128, [0.4,0.8], [64, 128], [[c_in, 128, 128, 256], [c_in, 128, 196, 256]])
        c_in = 512
        self.sa3 = PointnetSAModule([c_in, 256, 512, 1024], npoint=None, radius=None, nsample=None)
        c_in = 1536
        self.fp3 = PointnetFPModule([c_in, 256, 256])
        c_in = 576
        self.fp2 = PointnetFPModule([c_in, 256, 128])
        c_in = 128+input_channel_num+3
        self.fp1 = PointnetFPModule([c_in, 128, 128])

    def forward(self, xyz, feature=None):
        '''
        xyz:        [B, N, 3]
        feature:    [B, C, N]
        '''
        # Set Abstraction layers
        if feature is not None:
            l0_points = torch.cat([xyz.permute(0, 2, 1), feature], 1).contiguous()
        else:
            l0_points = xyz.permute(0, 2, 1).contiguous()
        l0_xyz = xyz.contiguous()        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([xyz.permute(0, 2, 1), l0_points], 1).contiguous(), l1_points)
        # F
        x = l0_points
        return x, l3_points

class pointnet2_seg_2(nn.Module):
    def __init__(self, input_channel_num=3):
        '''
        input_channel_num: 3+C
        '''
        super(pointnet2_seg_2, self).__init__()
        c_in = input_channel_num
        self.sa1 = PointnetSAModuleMSG(512, [0.1, 0.2, 0.4], [32, 64, 128], [[c_in, 32, 32, 64], [c_in, 64, 64, 128], [c_in, 64, 96, 128]])
        c_in = 128+128+64
        self.sa2 = PointnetSAModuleMSG(128, [0.4,0.8], [64, 128], [[c_in, 128, 128, 256], [c_in, 128, 196, 256]])
        c_in = 512
        self.sa3 = PointnetSAModule([c_in, 256, 512, 1024], npoint=None, radius=None, nsample=None)
        c_in = 1536
        self.fp3 = PointnetFPModule([c_in, 256, 256])
        c_in = 576
        self.fp2 = PointnetFPModule([c_in, 256, 128])
        c_in = 128+input_channel_num+3
        self.fp1 = PointnetFPModule([c_in, 128, 128])
        
        self.conv1 = basic_conv1d_seq([128, 128, 1])

    def forward(self, xyz, feature=None):
        '''
        xyz:        [B, N, 3]
        feature:    [B, C, N]
        '''
        # Set Abstraction layers
        if feature is not None:
            l0_points = torch.cat([xyz.permute(0, 2, 1), feature], 1).contiguous()
        else:
            l0_points = xyz.permute(0, 2, 1).contiguous()
        l0_xyz = xyz.contiguous()        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        x = self.conv1(l1_points)
        x = torch.sigmoid(x)
        object_gf = torch.max(x*l1_points, 2)[0]
        back_gf = torch.max((1-x)*l1_points, 2)[0]
        #l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([xyz.permute(0, 2, 1), l0_points], 1).contiguous(), l1_points)
        
        seg = x
        f = l1_points 
        dist, idx = three_nn(l0_xyz, l1_xyz)
        weight = torch.ones(idx.shape).to(idx.device)
        final_seg = three_interpolate(
            seg, idx, weight
        )
        return seg, l1_points, object_gf, back_gf, final_seg



class pointnet2_encoder(nn.Module):
    def __init__(self, input_channel_num=3, gf_channel_num=1024):
        '''
        input_channel_num: 3+C
        '''
        super(pointnet2_encoder, self).__init__()
        c_in = input_channel_num
        self.sa1 = PointnetSAModuleMSG(512, [0.1, 0.2, 0.4], [32, 64, 128], [[c_in, 32, 32, 64], [c_in, 64, 64, 128], [c_in, 64, 96, 128]])
        c_in = 128+128+64
        self.sa2 = PointnetSAModuleMSG(128, [0.4,0.8], [64, 128], [[c_in, 128, 128, 256], [c_in, 128, 196, 256]])
        c_in = 256+256
        self.sa3 = PointnetSAModule([c_in, 256, 512, gf_channel_num], npoint=None, radius=None, nsample=None)

    def forward(self, xyz, feature=None):
        '''
        xyz:        [B, N, 3]
        feature:    [B, C, N]
        '''
        # Set Abstraction layers
        if feature is not None:
            l0_points = torch.cat([xyz.permute(0, 2, 1), feature], 1).contiguous()
        else:
            l0_points = xyz.permute(0, 2, 1).contiguous()
        l0_xyz = xyz.contiguous()        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #l3_points = l3_points.squeeze()
        return l3_points

class pointnet2_encoder_test(nn.Module):
    def __init__(self, input_channel_num=3, gf_channel_num=1024):
        '''
        input_channel_num: 3+C
        '''
        super().__init__()
        c_in = input_channel_num
        self.sa1 = PointnetSAModuleMSG(512, [0.1, 0.2, 0.4], [32, 64, 128], [[c_in, 32, 32, 64], [c_in, 64, 64, 128], [c_in, 64, 96, 128]])
        c_in = 128+128+64
        self.sa2 = PointnetSAModuleMSG(128, [0.4,0.8], [64, 128], [[c_in, 128, 128, 256], [c_in, 128, 196, 256]])
        c_in = 256+256
        self.sa3 = PointnetSAModule_test([c_in, 256, 512, gf_channel_num], npoint=None, radius=None, nsample=None)

    def forward(self, xyz, feature=None):
        '''
        xyz:        [B, N, 3]
        feature:    [B, C, N]
        '''
        # Set Abstraction layers
        if feature is not None:
            l0_points = torch.cat([xyz.permute(0, 2, 1), feature], 1).contiguous()
        else:
            l0_points = xyz.permute(0, 2, 1).contiguous()
        l0_xyz = xyz.contiguous()        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #l3_points = l3_points.squeeze()
        return l3_points, l1_xyz, l1_points, l2_xyz, l2_points


class pointnet2_encoder2(nn.Module):
    def __init__(self, input_channel_num=3, gf_channel_num=1024):
        '''
        input_channel_num: 3+C
        '''
        super().__init__()
        c_in = input_channel_num
        self.sa1 = PointnetSAModuleMSG(512, [0.1, 0.2, 0.4], [32, 64, 128], [[c_in, 64, 96, 128], [c_in, 64, 96, 128], [c_in, 64, 96, 128]])
        c_in = 128+128+128
        self.sa2 = PointnetSAModuleMSG(128, [0.4,0.8], [64, 128], [[c_in, 128, 196, 256], [c_in, 128, 196, 256]])
        c_in = 256+256
        self.sa3 = PointnetSAModule([c_in, 512, 512, gf_channel_num], npoint=None, radius=None, nsample=None)

    def forward(self, xyz, feature=None):
        '''
        xyz:        [B, N, 3]
        feature:    [B, C, N]
        '''
        # Set Abstraction layers
        if feature is not None:
            l0_points = torch.cat([xyz.permute(0, 2, 1), feature], 1).contiguous()
        else:
            l0_points = xyz.permute(0, 2, 1).contiguous()
        l0_xyz = xyz.contiguous()        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #l3_points = l3_points.squeeze()
        return l3_points


def get_k_neighbor(points, k=8):
    '''
    points: [B, N, 3]   Tensor
    k:                  int
    '''
    x = points[:,:,:3]
    x1 = x.unsqueeze(1)
    x2 = x.unsqueeze(2)
    diff = (x1-x2).norm(dim=-1)
    dis, idx = diff.topk(k, largest=False)
    return idx.int()

def get_k_neighbor_2(points, k=8):
    '''
    points: [B, N, 3]   Tensor
    k:                  int
    '''
    x = points[:,:,:3]
    x1 = x.unsqueeze(1)
    x2 = x.unsqueeze(2)
    diff = (x1-x2).norm(dim=-1)
    dis, idx = diff.topk(k, largest=False)
    return idx.int(), dis

class PointConv(nn.Module):
    def __init__(self, input_channel_num, mlps, k=8, append_num=0):
        super(PointConv, self).__init__()
        self.k = k
        self.append_num = append_num
        self.mlp = MlpConv(input_channel_num*k+append_num, mlps)

    def forward(self, feature, idx, append_feature=None):
        '''
        feature:    [B, C, N] Tensor
        idx:        [B, N, K] int Tensor
        append_feature: [B, C2, N]
        '''
        B, C, N = feature.shape
        feature = feature.contiguous()
        idx = idx[:,:,:self.k].contiguous()
        grouped_feature = group(feature, idx)
        grouped_feature = grouped_feature.permute(0, 2, 3, 1)
        grouped_feature = grouped_feature.reshape(B, N, -1)
        grouped_feature = grouped_feature.permute(0, 2, 1)
        if append_feature is not None:
            grouped_feature = torch.cat([grouped_feature, append_feature], 1)
        res = self.mlp(grouped_feature)
        return res

class PointConvDis(nn.Module):
    def __init__(self, input_channel_num, mlps, k=8, append_num=0):
        super(PointConvDis, self).__init__()
        self.k = k
        self.append_num = append_num
        self.mlp = MlpConv(input_channel_num*k+k+append_num, mlps)

    def forward(self, feature, idx, dis, append_feature=None):
        '''
        feature:    [B, C, N] Tensor
        idx:        [B, N, K] int Tensor
        append_feature: [B, C2, N]
        dis:        [B, N, K]
        '''
        B, C, N = feature.shape
        feature = feature.contiguous()
        idx = idx[:,:,:self.k].contiguous()
        grouped_feature = group(feature, idx)
        grouped_feature = grouped_feature.permute(0, 2, 3, 1)
        dis = dis[:,:,:self.k].unsqueeze(3)
        #print(dis.shape, grouped_feature.shape)
        grouped_feature = torch.cat([dis, grouped_feature], 3)
        grouped_feature = grouped_feature.reshape(B, N, -1)
        grouped_feature = grouped_feature.permute(0, 2, 1)
        if append_feature is not None:
            grouped_feature = torch.cat([grouped_feature, append_feature], 1)
        res = self.mlp(grouped_feature)
        return res

class PointConvAttention(nn.Module):
    def __init__(self, input_channel_num, mlps, k=8, append_num=0):
        super(PointConvAttention, self).__init__()
        self.k = k
        self.append_num = append_num
        self.mlp = MlpConv(input_channel_num*k+append_num, mlps)
        self.weight_mlp = MlpConv(input_channel_num*k, [128, 128, k])
        #self.attention_mlp = MlpConv(input_channel_num+1, [256, input_channel_num])

    def forward(self, feature, idx, append_feature=None):
        '''
        feature:    [B, C, N] Tensor
        idx:        [B, N, K] int Tensor
        append_feature: [B, C2, N]
        '''
        B, C, N = feature.shape
        feature = feature.contiguous()
        idx = idx[:,:,:self.k].contiguous()
        grouped_feature = group(feature, idx)
        grouped_feature = grouped_feature.permute(0, 2, 3, 1)
        grouped_feature_n = grouped_feature
        grouped_feature = grouped_feature.reshape(B, N, -1).permute(0, 2, 1)

        weight = self.weight_mlp(grouped_feature).permute(0, 2, 1).unsqueeze(3)
        weight = F.softmax(weight, 2)

        grouped_feature_n = grouped_feature_n*weight
        grouped_feature_n = grouped_feature_n.reshape(B, N, -1).permute(0, 2, 1)

        if append_feature is not None:
            grouped_feature_n = torch.cat([grouped_feature_n, append_feature], 1)
        res = self.mlp(grouped_feature_n)
        return res

class PointConv2(nn.Module):
    def __init__(self, input_channel_num, mlps, k=8, append_num=0):
        super(PointConv2, self).__init__()
        self.k = k
        self.append_num = append_num
        self.mlp1 = MlpConv(input_channel_num*k+append_num, [512, k*64])
        self.mlp2 = MlpConv(input_channel_num+append_num, [128, 64])
        self.mlp3 = MlpConv(k*64, mlps)

    def forward(self, feature, idx, append_feature=None):
        '''
        feature:    [B, C, N] Tensor
        idx:        [B, N, K] int Tensor
        append_feature: [B, C2, N]
        '''
        B, C, N = feature.shape
        feature = feature.contiguous()
        idx = idx[:,:,:self.k].contiguous()
        
        grouped_feature = group(feature, idx)
        grouped_feature = grouped_feature.permute(0, 2, 3, 1)
        grouped_feature = grouped_feature.reshape(B, N, -1)
        grouped_feature = grouped_feature.permute(0, 2, 1)
        if append_feature is not None:
            grouped_feature = torch.cat([grouped_feature, append_feature], 1)
        L = self.mlp1(grouped_feature)
        
        if append_feature is not None:
            feature2 = torch.cat([feature, append_feature], 1)
        else:
            feature2 = feature
        G = self.mlp2(feature2)
        G = group(G, idx)
        G = G.permute(0, 2, 3, 1)
        G = G.reshape(B, N, -1)
        G = G.permute(0, 2, 1)

        res = self.mlp3(G+L)
        return res


if __name__ == '__main__1':
    net = pointnet2_seg(3+1024) 
    net.to('cuda')
    pc = torch.randn(4, 2048, 3).cuda()
    feature = torch.randn(4, 1024, 2048).cuda()
    out = net(pc, feature)

if __name__ == '__main__':
    points = torch.randn(4, 2048, 3).cuda()
    feature = torch.randn(4, 256, 2048).cuda()
    idx = get_k_neighbor(points)
    print(idx.shape)
    model = PointConvAttention(256, [512, 128], 4,  256).cuda()
    res = model(feature, idx, feature)
    print(res.shape)
    input()
