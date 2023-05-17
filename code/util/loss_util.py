import os
import sys

import torch
import torch.nn as nn

import numpy as np

from chamfer_distance import ChamferDistance, ChamferDistanceIdx

global_cd = ChamferDistance()
global_cdi = ChamferDistanceIdx()

def replace_nan(a):
    a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    return a


def seg_loss(x, gt_seg):
    d1 = torch.clamp(x+1e-7, 0, 1) 
    d2 = torch.clamp(1-x+1e-7, 0, 1)
    seg = -(gt_seg*torch.log(d1)+(1-gt_seg)*torch.log(d2))
    seg = seg[:,:,0]
    seg = torch.mean(seg, 1)
    return seg


# Basic virtual class, DO NOT use this 
class BasicLoss(nn.Module):
    def __init__(self):
        super(BasicLoss, self).__init__()
        self.loss_num = 0
        self.loss_name = []
    
    def batch_forward(self, outputs, gts):
        return []

    def forward(self, outputs, gts):
        loss = self.batch_forward(outputs, gts)
        for i in range(len(loss)):
            loss[i] = torch.mean(loss[i])
        return loss


class ChamferLoss(BasicLoss):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.distance = ChamferDistance()
        self.loss_num = 3
        self.loss_name = ['loss', 'p2g', 'g2p']
    
    def batch_forward(self, points, gts):
        p2g, g2p = self.distance(points, gts)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        loss = p2g + g2p
        return [loss, p2g, g2p]

    '''
    shape: [bs, point num, 3]
    
    def forward(self, points, gts):
        loss = self.batch_forward(points, gts)
        for i in range(len(loss)):
            loss[i] = torch.mean(loss[i])
        return loss
    '''

class ChamferLossSqrt(nn.Module):
    def __init__(self):
        super(ChamferLossSqrt, self).__init__()
        self.distance = ChamferDistance()
        self.loss_num = 3
        self.loss_name = ['loss', 'p2g', 'g2p']
    
    '''
    shape: [bs, point num, 3]
    '''
    def batch_forward(self, points, gts):
        p2g, g2p = self.distance(points, gts)
        p2g = torch.mean(p2g, 1)
        p2g = torch.sqrt(p2g)
        g2p = torch.mean(g2p, 1)
        g2p = torch.sqrt(g2p)
        loss = (p2g + g2p) / 2
        return [loss, p2g, g2p]
    
    def forward(self, points, gts):
        loss = self.batch_forward(points, gts)
        for i in range(len(loss)):
            loss[i] = torch.mean(loss[i])
        return loss

class ChamferLossSqrtT(nn.Module):
    def __init__(self):
        super(ChamferLossSqrtT, self).__init__()
        self.distance = ChamferDistance()
        self.loss_num = 3
        self.loss_name = ['loss', 'p2g', 'g2p']
    
    '''
    shape: [bs, point num, 3]
    '''
    def batch_forward(self, points, gts):
        p2g, g2p = self.distance(points, gts)
        p2g = torch.sqrt(p2g)
        p2g = torch.mean(p2g, 1)
        g2p = torch.sqrt(g2p)
        g2p = torch.mean(g2p, 1)
        
        loss = p2g + g2p
        return [loss, p2g, g2p]
    
    def forward(self, points, gts):
        loss = self.batch_forward(points, gts)
        for i in range(len(loss)):
            loss[i] = torch.mean(loss[i])
        return loss

class PcnLoss(nn.Module):
    def __init__(self, criterion, alpha=1.0):
        super(PcnLoss, self).__init__()
        self.distance = criterion
        self.alpha = alpha
        self.loss_num = 3
        self.loss_name = ['total_loss', 'coarse_loss', 'fine_loss']
    
    '''
    shape: [bs, point num, 3]
    '''
    def batch_forward(self, points, gts):
        coarse, fine = points
        loss_coarse, _, _ = self.distance.batch_forward(coarse, gts)
        loss_fine, _, _ = self.distance.batch_forward(fine, gts)
        loss_tot = loss_coarse + self.alpha * loss_fine
        return [loss_tot, loss_coarse, loss_fine]
    
    def forward(self, points, gts):
        loss = self.batch_forward(points, gts)
        for i in range(len(loss)):
            loss[i] = torch.mean(loss[i])
        return loss


class PointSegLoss(BasicLoss):
    def __init__(self):
        super(PointSegLoss, self).__init__()
        self.loss_num = 1
        self.loss_name = ['seg_loss']

    def batch_forward(self, x, gt):
        loss = torch.log(x+1e-7)
        loss = gt*loss
        loss = -torch.sum(loss, 2)
        loss = torch.mean(loss, 1)
        return loss


class SmoothLoss(BasicLoss):
    def __init__(self, K=4):
        super(SmoothLoss, self).__init__()
        self.K = K+1
        self.loss_num = 1
        self.loss_name = ['smooth_loss']

    def batch_forward(self, x, gts):
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(2)
        diff = (x1-x2).norm(dim=-1)
        diff, idx = diff.topk(self.K, largest=False)
        # print(idx.shape)
        loss = diff[:,:,1:].mean(2).std(1)
        return [loss]


def acc_for_seg(x, y):
    '''
    x : [B, N, 1]
    y : [B, N, 1]
    can NOT backward, only for test
    '''
    dev = x.device
    B, N, _ = x.shape
    x = x.detach().cpu().numpy()[:,:,0]
    y = y.detach().cpu().numpy()[:,:,0]
    x = (x>0.5)*1
    x = x.astype(np.int32)
    y = y.astype(np.int32)  
    res = 1 - ((x+y) == 1) * 1
    res = np.sum(res, axis=1).astype(np.float32)
    res = res/N*100
    res = torch.from_numpy(res).to(dev)
    #print(res.shape)
    #print(res)
    return res


def F1_for_seg(x, y):
    '''
    x : [B, N]
    y : [B, N]
    can NOT backward, only for test
    '''
    x = x.int()
    y = y.int()
    t1 = x+y
    t2 = x-y
    #TN = torch.sum((t1==0).float(), 1)
    TP = torch.sum((t1==2).float(), 1)
    FN = torch.sum((t2==-1).float(), 1)
    FP = torch.sum((t2==1).float(), 1)
    pre = TP/(TP+FP)
    pre = replace_nan(pre)
    rec = TP/(TP+FN)
    rec = replace_nan(rec)
    F1 = 2*(pre*rec)/(pre+rec)
    F1 = replace_nan(F1)
    return F1


def IOU_for_seg(x, y):
    '''
    x : [B, N]
    y : [B, N]
    can NOT backward, only for test
    '''
    x = x.int()
    y = y.int()
    iou = (torch.sum(torch.logical_and(x, y), 1).float()) / (torch.sum(torch.logical_or(x, y), 1).float())
    return iou

def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

# F-Score
def calc_fcd(points, gt, a=0.0001):
    #print(gt)
    dist1, dist2 = global_cd(points, gt)
    f1, _, _ = fscore(dist1, dist2, a)
    return f1


if __name__ == '__main__1':
    from io_util import *
    cls = PcnLoss(ChamferLossSqrt()).to('cuda')
    points = read_point_cloud('/data/mcf/new_voxel_net/code/topnet/gts_pc.txt')
    points = torch.from_numpy(points).to('cuda')
    points = torch.unsqueeze(points, 0)
    print(points.shape)
    gts = read_point_cloud('/data/mcf/new_voxel_net/code/topnet/res_pc.txt')
    gts = torch.from_numpy(gts).to('cuda')
    gts = torch.unsqueeze(gts, 0)
    print(gts.shape)
    points = [points, points]
    print(cls(points, gts)) # 0.018135
    l, a, b = cls.batch_forward(points, gts)
    print(l, a, b)

if __name__ == '__main__':
    print('test smooth loss')
    pc = torch.randn(4, 2048, 3).cuda()
    loss = SmoothLoss(4)
    print('rand points', loss(pc, None))
    from io_util import *
    gts = read_point_cloud('/home/mcf/data/works/sc/code/topnet2_seg_pn2/log_chair/res/0/gt_points_pc.txt')
    gts = torch.from_numpy(gts).to('cuda')
    gts = torch.unsqueeze(gts, 0)
    print('gts points', loss(gts, None))
    res = read_point_cloud('/home/mcf/data/works/sc/code/topnet2_seg_pn2/log_chair/res/0/res_points_pc.txt')
    res = torch.from_numpy(res).to('cuda')
    res = torch.unsqueeze(res, 0)
    print('res points', loss(res, None))

