import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    cuda_index = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append('../util')
sys.path.append('..')
from loss_util import *
from point_util import *
from base_model_util import *
import pointnet2_model_api as PN2
from pointnet2_ops.pointnet2_utils import QueryAndGroup
# from avg_shape_2.avg_shape_1 import Model as Model_WSLoss


class USSPA_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.GPN = 512
        self.E_R = PcnEncoder2(out_c=512)
        self.E_A = PcnEncoder2(out_c=512)
        self.D_R = MlpConv(512, [512, 512, 1024, 1024, self.GPN*3])
        self.D_A = MlpConv(512, [512, 512, 1024, 1024, self.GPN*3])

        self.mlp_mirror_ab = MlpConv(512, [128, 128, 2])

        self.mlp_refine_1 = MlpConv(3, [256, 256, 256])
        self.mlp_refine_2 = MlpConv(512, [256, 256, 256])
        self.qg = QueryAndGroup(0.25, 32)
        self.mlp_refine_3 = MlpConv(512, [256, 256, 256])

        self.UPN = 4
        self.mlp_refine_4 = MlpConv(256+256+3, [512, 512, 3*self.UPN])

    
    def get_mirror(self, point, ab):
        __e = 1e-8
        A, B = torch.split(ab, [1, 1], 1)

        x = point[:,:,0:1]
        z = point[:,:,2:3]

        AxBz = 2*(A*x+B*z)/(A**2+B**2+__e)

        new_x = x - A*AxBz
        new_z = z - B*AxBz

        y = point[:,:,1:2]
        point = torch.cat([new_x, y, new_z], 2)
        return point


    def upsampling_refine(self, point):
        #### encode feature ####
        B, N, _ = point.shape
        x = self.mlp_refine_1(point.permute(0, 2, 1))
        x_max = torch.max(x, 2, keepdim=True).values
        x = self.mlp_refine_2(torch.cat([x, x_max.repeat([1, 1, N])], 1))
        x_local = self.qg(point, point, x)[:,3:,:,:]        # [B, 256, 512, 32]
        x_local = torch.max(x_local, -1).values                           # [B, 256, 512]
        x = self.mlp_refine_3(torch.cat([x, x_local], 1))
        x_max = torch.max(x, -1, keepdim=True).values

        #### upsampling refine ####      
        x = torch.cat([point.permute(0, 2, 1), x, x_max.repeat([1, 1, N])], 1)
        shift = self.mlp_refine_4(x)

        #### shift ####
        res = torch.unsqueeze(point, 2).repeat([1, 1, self.UPN, 1])
        res = torch.reshape(res, [B, -1, 3])
        shift = shift.permute(0, 2, 1).reshape([B, -1, 3])
        res = res + shift

        return res
    
    def mi_sam(self, point, ab):
        N = point.shape[1]
        point_M = self.get_mirror(point, ab)
        point = torch.cat([point, point_M], 1)
        point = PN2.FPS(point, N)
        return point

    def forward(self, input_R, input_A):
        B, N, _ = input_R.shape

        f_R_0 = self.E_R(input_R)
        point_R_0 = self.D_R(f_R_0)
        point_R_0 = point_R_0.reshape([-1, self.GPN, 3])

        ab = torch.tanh(self.mlp_mirror_ab(f_R_0))
        input_R_M = self.get_mirror(input_R, ab)

        input_R_point_R_0 = torch.cat([input_R, input_R_M, point_R_0], 1)
        input_R_point_R_0 = PN2.FPS(input_R_point_R_0, 2048)

        #### autoencoding ####

        x = torch.cat([input_R_point_R_0, input_A], 0)
        x = self.E_A(x)
        f_R, f_A = torch.split(x, [B, B], 0)
        f = x
        x = self.D_A(x)
        x = x.reshape([-1, self.GPN, 3])
        point_R, point_A = torch.split(x, [B, B], 0)

        x = self.upsampling_refine(x)
        point_R_3, point_A_3 = torch.split(x, [B, B], 0)
        point_R_3 = self.mi_sam(point_R_3, ab)

        other = []
        other.append(point_R_0)
        other.append(input_R_M)

        return f_R, f_A, point_R, point_R_3, point_A, point_A_3, input_R_point_R_0, other


class PointDIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PcnEncoder2(out_c=256)
        self.mlp = MlpConv(256, [64, 64, 1])
    
    def forward(self, point):
        d_p = self.encoder(point)
        d_p = self.mlp(d_p)
        d_p = torch.sigmoid(d_p)
        d_p = d_p[:,0,0]
        return d_p


class USSPA_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_f = MlpConv(512, [64, 64, 1])
        self.d_p = PointDIS()
    
    def discriminate_feature(self, f):
        d_f = self.d_f(f)
        d_f = torch.sigmoid(d_f)
        d_f = d_f[:,0,0]
        return d_f
    
    def forward(self, f_R, f_A, input_R_point_R_0, point_R_3, input_A):
        B = f_R.shape[0]
        f = torch.cat([f_R, f_A], 0)
        d_f = self.discriminate_feature(f)
        d_f_R, d_f_A = torch.split(d_f, [B, B], 0)
        point = torch.cat([input_R_point_R_0, point_R_3, input_A], 0)
        d_p = self.d_p(point)
        d_p_R, d_p_R_3, d_p_A = torch.split(d_p, [B, B, B], 0)

        return d_f_R, d_f_A, d_p_R, d_p_R_3, d_p_A


class USSPA(nn.Module):
    def __init__(self, dis = 0.03):
        super().__init__()
        self.G = USSPA_G()

        self.D = USSPA_D()
        self.loss = USSPALoss()
        self.loss_test = USSPALoss_test()
    
    def forward(self, data):
        rc_data, sn_data = data
        input_R = rc_data[0]
        input_A = sn_data[0]

        f_R, f_A, point_R, point_R_3, point_A, point_A_3, input_R_point_R_0, other = self.G(input_R, input_A)
        d_f_R, d_f_A, d_p_R, d_p_R_3, d_p_A = self.D(f_R, f_A, input_R_point_R_0, point_R_3, input_A)

        return point_R, point_R_3, point_A, point_A_3, input_R_point_R_0, \
            d_f_R, d_f_A, d_p_R, d_p_R_3, d_p_A, \
            input_R, input_A, other


class USSPALoss(BasicLoss):
    def __init__(self):
        super().__init__()
        self.loss_name = ['loss_g', 'loss_d', 'g_fake_loss', 'g_rsl_2', 'g_rsl_2', 'g_fsl_3', 'density_loss', 'd_fake_loss', 'd_real_loss']
        self.loss_num = len(self.loss_name)
        self.distance = ChamferDistance()
    
    def cd(self, p1, p2):
        p2g, g2p = self.distance(p1, p2)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        cd = p2g + g2p
        return cd, p2g, g2p
    
    def density_loss(self, x):
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(2)
        diff = (x1-x2).norm(dim=-1)
        diff, idx = diff.topk(16, largest=False)
        # print(idx.shape)
        loss = diff[:,:,1:].mean(2).std(1)
        return loss
    
    def batch_forward(self, outputs, data):
        __E = 1e-8
        point_R, point_R_3, point_A, point_A_3, input_R_point_R_0, \
        d_f_R, d_f_A, d_p_R, d_p_R_3, d_p_A, \
        input_R, input_A, other = outputs

        point_R_0 = other[0]

        g_fake_loss = -torch.log(d_f_R+__E) -torch.log(d_p_R+__E) -torch.log(d_p_R_3+__E)

        g_rsl, _, _ = self.cd(point_A, input_A)
        g_rsl_2, _, _ = self.cd(point_A_3, input_A)
        g_fsl, _, _ = self.cd(point_R, input_R_point_R_0)
        g_fsl_2, _, _ = self.cd(point_R_3, input_R_point_R_0)
        _, _, g_fsl_3 = self.cd(point_R_0, input_R)

        density_loss = self.density_loss(point_A) + self.density_loss(point_R)

        loss_g = g_fake_loss + 1e2 * g_rsl + 1e2 * g_rsl_2 + 1e2 * g_fsl + 1e2 * g_fsl_2 + 1e2 * g_fsl_3 + 7.5 * density_loss

        d_fake_loss = -torch.log(1-d_f_R+__E) -torch.log(1-d_p_R+__E) -torch.log(1-d_p_R_3+__E)
        d_real_loss = -torch.log(d_f_A+__E) -torch.log(d_p_A+__E)
        loss_d = (d_real_loss+d_fake_loss)/2

        return [loss_g, loss_d, g_fake_loss, g_rsl_2, g_fsl_2, g_fsl_3, density_loss, d_fake_loss, d_real_loss]


class USSPALoss_test(BasicLoss):
    def __init__(self):
        super().__init__()
        self.loss_name = ['cd', 'fcd_0p001', 'fcd_0p01', 'den_loss']
        self.loss_num = len(self.loss_name)
        self.distance = ChamferDistance()
    
    def cd1(self, p1, p2):
        p2g, g2p = self.distance(p1, p2)
        p2g = torch.sqrt(p2g)
        g2p = torch.sqrt(g2p)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        cd = (p2g + g2p)/2
        return cd

    def cd2(self, p1, p2):
        p2g, g2p = self.distance(p1, p2)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        cd = p2g + g2p
        return cd

    def density_loss(self, x):
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(2)
        diff = (x1-x2).norm(dim=-1)
        diff, idx = diff.topk(16, largest=False)
        loss = diff[:,:,1:].mean(2)
        mean = loss.mean(1)
        loss = loss.std(1)
        return loss, mean
    

    def batch_forward(self, outputs, data):
        __E = 1e-8
        point_R, point_R_3, point_A, point_A_3, input_R_point_R_0, \
        d_f_R, d_f_A, d_p_R, d_p_R_3, d_p_A, \
        input_R, input_A, other = outputs

        gt = data[0][1]

        cd = self.cd1(point_R_3, gt)
        fcd_0p001 = calc_fcd(point_R_3, gt, a=0.001)
        fcd_0p01 = calc_fcd(point_R_3, gt, a=0.01)
        
        den_loss, mean = self.density_loss(point_R_3)

        return [cd, fcd_0p001, fcd_0p01, den_loss]    

if __name__ == '__main__':
    model = USSPA()