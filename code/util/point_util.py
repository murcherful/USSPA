import os
import sys
import numpy as np
import torch

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


def resample_pcd_idx(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    idx = idx[:n]
    return pcd[idx], idx



def get_sxyz_of_pc(pc):
    maxx = abs(np.max(pc[:,0]))
    minx = abs(np.min(pc[:,0]))
    maxy = abs(np.max(pc[:,1]))
    miny = abs(np.min(pc[:,1]))
    maxz = abs(np.max(pc[:,2]))
    minz = abs(np.min(pc[:,2]))
    return np.array([max(maxx, minx), max(maxy, miny), max(maxz, minz)], np.float32)


def get_sxyz_of_pcs(pcs):
    return np.array([get_sxyz_of_pc(pc) for pc in pcs], np.float32)


def voxelize_std_pcs_torch(pcs, voxel_size):
    pcs = pcs + 1
    pcs *= (voxel_size/2)
    pcs = pcs.int()
    pcs = torch.clamp(pcs, 0, voxel_size-1)
    return pcs

def get_index_of_std_pcs_torch(pcs, voxel_size):
    pcs_std = voxelize_std_pcs_torch(pcs, voxel_size)
    pcs_std[:,0,:] *= voxel_size*voxel_size
    pcs_std[:,1,:] *= voxel_size
    index = torch.sum(pcs_std, dim=1)
    return index

def voxelize_std_pc_np(pcs, voxel_size):
    pcs = pcs + 1
    pcs *= (voxel_size/2)
    pcs = pcs.astype(np.int)
    pcs = np.clip(pcs, 0, voxel_size-1)
    return pcs

def get_index_of_std_pc_np(pcs, voxel_size):
    pcs_std = voxelize_std_pc_np(pcs, voxel_size)
    pcs_std[:,0] *= voxel_size*voxel_size
    pcs_std[:,1] *= voxel_size
    index = np.sum(pcs_std, 1, keepdims=True)
    return index


def group_index(index):
    num = index.shape[0]
    level_num = index.shape[1]
    ds = []
    max_num = []
    for k in range(level_num):
        ds.append({})
        max_num.append(0)
    for j in range(num):
        for k in range(level_num):
            _idx = index[j,k]
            if _idx not in ds[k]:
                ds[k][_idx] = []
            ds[k][_idx].append(j) 
            max_num[k] = max(len(ds[k][_idx]), max_num[k])
    return ds, max_num


def group_indexes(indexes):
    res = []
    max_num_s = []
    batch_size = indexes.shape[0]
    for i in range(batch_size):
        ds, max_num = group_index(indexes[i])
        res.append(ds)
        max_num_s.append(max_num)
    return res, max_num_s


def get_octree_index_np(pc, level):
    voxel_size = 2**level
    pc_voxeled = voxelize_std_pc_np(pc, voxel_size)
    res = np.zeros([pc_voxeled.shape[0]])
    for i in range(level):
        voxel_size /= 2
        rr = pc_voxeled // voxel_size
        mm = np.mod(pc_voxeled, voxel_size)
        res += (voxel_size)**3*(4*rr[:,0]+2*rr[:,1]+rr[:,2])
        pc_voxeled = mm
    res = np.expand_dims(res, 1).astype(np.int)
    return res
    

def get_split_num(indexes):
    res = []
    res2 = []
    for i in range(indexes.shape[1]):
        num = np.bincount(indexes[:,i])
        index = np.where(num!=0)
        res2.append(index[0])
        num = num[index]
        res.append(num)
    return res, res2

def normalize(pcd, get_arg=False, center=None, max_scale=None):
    if center is None or max_scale is None:
        maxs = np.max(pcd, 0, keepdims=True)
        mins = np.min(pcd, 0, keepdims=True)
        center = (maxs+mins)/2
        scale = (maxs-mins)/2
        max_scale = np.max(scale)
    pcd = pcd - center
    pcd = pcd / max_scale
    if get_arg:
        return pcd, center, max_scale
    else:  
        return pcd