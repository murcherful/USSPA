import os 
import sys 

import torch
from torch.utils.data import Dataset, DataLoader

import lmdb

import numpy as np

import msgpack
import msgpack_numpy   
msgpack_numpy.patch()

from print_util import *
from point_util import *
from io_util import *

'''
pip install lmdb
pip install msgpack-numpy
'''


# virtual class, DO NOT use
class MyDataset(Dataset):
    def __init__(self, lmdb_path, prefix='[MYDARASET]'):
        self.lmdb_path = lmdb_path
        self.prefix = prefix
        self.have_self_collate_fn = False

        # load lmdb
        self.env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, map_size=1099511627776 * 2)
        self.db = self.env.begin()

        print_info('open db: ' + self.lmdb_path, prefix=self.prefix)
        # load keys
        self.keys = []
        keys = self.db.get(b'__keys__')
        if keys is not None:
            self.keys = msgpack.loads(keys, raw=False)
        else:
            for k in self.db.cursor(): 
                self.keys.append(k[0])
        self.size = len(self.keys)
        print_info('get %d entries' % self.size, prefix=self.prefix)

    def __del__(self):
        self.env.close()
        print_info('close db: ' + self.lmdb_path, prefix=self.prefix)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        buf = self.db.get(self.keys[index])
        data = msgpack.loads(buf, raw=False)
        return data

    @staticmethod 
    def to_device(data, device):
        # data = data.to(device)
        return None 
    
    @staticmethod
    def _collate_fn(datas):
        # datas = torch.from_numpy(datas)
        return None
    

class RealComDataset(MyDataset):
    def __init__(self, lmdb_path, input_pn, gt_pn, class_name='all', prefix='[REALCOMDARASET]'):
        super().__init__(lmdb_path, prefix)
        self.input_pn = input_pn
        self.gt_pn = gt_pn
        self.class_dict = {
            'chair'             :   0,    
            'table'             :   1,    
            'trash_bin'         :   2,    
            'tv_or_monitor'     :   3,        
            'cabinet'           :   4,
            'bookshelf'         :   5,
            'sofa'              :   6,
            'lamp'              :   7,
            'bed'               :   8,    
            'tub'               :   9,    
        }
        if class_name != 'all':
            id_dict = self.db.get(b'__id_dict__')
            assert id_dict is not None 
            self.id_dict = msgpack.loads(id_dict, raw=False)
            selected_keys = []
            for key in self.id_dict.keys():
                class_id = key.split('-')[1]
                if class_id == class_name:
                    idx = self.id_dict[key]
                    selected_keys.append(str(idx).encode())
            self.keys = selected_keys
            self.size = len(self.keys)
            print_info('select %d entries with class: %s' % (self.size, class_name), prefix=self.prefix)

    # some real point clouds are too large to train, which are not correct for real objects
    # we clip them into [-1, 1]^3
    def clip_points(self, points):
        idx = np.ones_like(points[:, 0])
        for i in range(3):
            idx1 = points[:, i] >= -1.0
            idx2 = points[:, i] <= 1.0
            idx = np.logical_and(idx, idx1)
            idx = np.logical_and(idx, idx2)
        points = points[idx,:]
        return points

    def __getitem__(self, index):
        data = super().__getitem__(index)
        data_id, incomplete_pcd, complete_pcd, model_T, model_R, model_S = data
        class_id = data_id.split('-')[1]
        complete_pcd, center, max_scale = normalize(complete_pcd, get_arg=True)
        incomplete_pcd = normalize(incomplete_pcd, center=center, max_scale=max_scale)

        incomplete_pcd = self.clip_points(incomplete_pcd)
        # some real point clouds are not match with gt after clip they contains 0 point
        # we repalce real point clouds with gt point clouds   
        if incomplete_pcd.shape[0] <= 10:
            incomplete_pcd = complete_pcd

        complete_pcd = resample_pcd(complete_pcd, self.gt_pn)
        incomplete_pcd = resample_pcd(incomplete_pcd, self.input_pn)

        return incomplete_pcd, complete_pcd, class_id
    
    @staticmethod
    def to_device(data, device):
        data = list(data)
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        return data

    @staticmethod
    def _collate_fn(datas):
        _incom_points = torch.from_numpy(np.array([data[0] for data in datas]).astype(np.float32))
        _com_points = torch.from_numpy(np.array([data[1] for data in datas]).astype(np.float32))
        _ids = [data[2] for data in datas]
        return _incom_points, _com_points, _ids 
    
        
class ShapeNetDataset(MyDataset):
    def __init__(self, lmdb_path, gt_pn, class_name='all', prefix='[SHAPENETDARASET]'):
        super().__init__(lmdb_path, prefix)
        self.gt_pn = gt_pn
        self.class_dict = {
            'chair'             :   0,    
            'table'             :   1,    
            'trash_bin'         :   2,    
            'tv_or_monitor'     :   3,        
            'cabinet'           :   4,
            'bookshelf'         :   5,
            'sofa'              :   6,
            'lamp'              :   7,
            'bed'               :   8,    
            'tub'               :   9,    
        }
        if class_name != 'all':
            id_dict = self.db.get(b'__id_dict__')
            assert id_dict is not None 
            self.id_dict = msgpack.loads(id_dict, raw=False)
            selected_keys = []
            __l = len(class_name)
            for key in self.id_dict.keys():
                if key[:__l] == class_name:
                    idx = self.id_dict[key]
                    selected_keys.append(str(idx).encode())
            self.keys = selected_keys
            self.size = len(self.keys)
            print_info('select %d entries with class: %s' % (self.size, class_name), prefix=self.prefix)
    
    def split_class(self, data_id):
        x = data_id.split('_')[:-1]
        res = x[0]
        for _x in x[1:]:
            res = res + '_' + _x 
        return res

    def __getitem__(self, index):
        data = super().__getitem__(index)
        data_id, points = data 
        # print(data_id)
        class_id = self.split_class(data_id)
        points = normalize(points)
        points = resample_pcd(points, self.gt_pn)
        return points, class_id, data_id
    
    @staticmethod
    def to_device(data, device):
        data = list(data)
        data[0] = data[0].to(device)
        return data

    @staticmethod
    def _collate_fn(datas):
        _points = torch.from_numpy(np.array([data[0] for data in datas]).astype(np.float32))
        _ids = [data[1] for data in datas]
        return _points, _ids 


class RealComGANDataset(Dataset):
    def __init__(self, lmdb_realcom_path, lmdb_sn_path, input_pn, gt_pn, class_name='all'):
        super().__init__()
        # print_info('Combine two datasets', prefix='[RealComGAN]')
        self.rc = RealComDataset(lmdb_realcom_path, input_pn, gt_pn, class_name)
        self.sn = ShapeNetDataset(lmdb_sn_path, gt_pn, class_name)
        self.sn_len = len(self.sn)
        self.sn_idx = self.sn_len
        self.sn_index = list(range(self.sn_len))
        np.random.shuffle(self.sn_index)
        self.class_dict = {
            'chair'             :   0,    
            'table'             :   1,    
            'trash_bin'         :   2,    
            'tv_or_monitor'     :   3,        
            'cabinet'           :   4,
            'bookshelf'         :   5,
            'sofa'              :   6,
            'lamp'              :   7,
            'bed'               :   8,    
            'tub'               :   9,    
        }
    
    def get_sn_index(self):
        if self.sn_idx >= self.sn_len:
            self.sn_idx = 0
            np.random.shuffle(self.sn_index)
        # print('sn index', self.sn_idx)
        res = self.sn_index[self.sn_idx]
        # print('shapenet index', res)
        self.sn_idx += 1
        return res        
    
    def __len__(self):
        return len(self.rc)

    def __getitem__(self, index):
        rc_data = self.rc.__getitem__(index)
        sn_data = self.sn.__getitem__(self.get_sn_index())
        return rc_data, sn_data
    
    @staticmethod
    def to_device(data, device):
        rc_data, sn_data = data
        rc_data = RealComDataset.to_device(rc_data, device) 
        sn_data = ShapeNetDataset.to_device(sn_data, device)
        return rc_data, sn_data

    @staticmethod
    def _collate_fn(datas):
        # fix here, datas = [[w, s], [w, s], ...]
        rc_datas = [data[0]for data in datas]
        sn_datas = [data[1]for data in datas]
        rc_datas = RealComDataset._collate_fn(rc_datas)
        sn_datas = ShapeNetDataset._collate_fn(sn_datas)
        return rc_datas, sn_datas


class RealComGANDataset_noise_rob(Dataset):
    def __init__(self, lmdb_realcom_path, lmdb_sn_path, input_pn, gt_pn, noise_scale=0.1, class_name='all'):
        super().__init__()
        self.noise_scale = noise_scale
        print_info('noise scale %f' % noise_scale)
        # print_info('Combine two datasets', prefix='[RealComGAN]')
        self.rc = RealComDataset(lmdb_realcom_path, input_pn, gt_pn, class_name)
        self.sn = ShapeNetDataset(lmdb_sn_path, gt_pn, class_name)
        self.sn_len = len(self.sn)
        self.sn_idx = self.sn_len
        self.sn_index = list(range(self.sn_len))
        np.random.shuffle(self.sn_index)
        self.class_dict = {
            'chair'             :   0,    
            'table'             :   1,    
            'trash_bin'         :   2,    
            'tv_or_monitor'     :   3,        
            'cabinet'           :   4,
            'bookshelf'         :   5,
            'sofa'              :   6,
            'lamp'              :   7,
            'bed'               :   8,    
            'tub'               :   9,    
        }
    
    def get_sn_index(self):
        if self.sn_idx >= self.sn_len:
            self.sn_idx = 0
            np.random.shuffle(self.sn_index)
        # print('sn index', self.sn_idx)
        res = self.sn_index[self.sn_idx]
        # print('shapenet index', res)
        self.sn_idx += 1
        return res        
    
    def __len__(self):
        return len(self.rc)

    def __getitem__(self, index):
        rc_data = self.rc.__getitem__(index)
        
        incomplete_pcd, complete_pcd, class_id = rc_data
        noise = np.random.random(incomplete_pcd.shape)
        noise = noise*2 - 1
        noise = noise * self.noise_scale
        incomplete_pcd = incomplete_pcd + noise
        rc_data = [incomplete_pcd, complete_pcd, class_id]

        sn_data = self.sn.__getitem__(self.get_sn_index())
        return rc_data, sn_data
    
    @staticmethod
    def to_device(data, device):
        rc_data, sn_data = data
        rc_data = RealComDataset.to_device(rc_data, device) 
        sn_data = ShapeNetDataset.to_device(sn_data, device)
        return rc_data, sn_data

    @staticmethod
    def _collate_fn(datas):
        # fix here, datas = [[w, s], [w, s], ...]
        rc_datas = [data[0]for data in datas]
        sn_datas = [data[1]for data in datas]
        rc_datas = RealComDataset._collate_fn(rc_datas)
        sn_datas = ShapeNetDataset._collate_fn(sn_datas)
        return rc_datas, sn_datas


class PcnDataset(Dataset):
    def __init__(self, path, input_pn, gt_pn, view_num=1, class_name='all', prefix='[PCNDARASET]'):
        super().__init__()
        self.path = path
        self.prefix = prefix
        self.class_name = class_name
        self.input_pn = input_pn
        self.gt_pn = gt_pn 
        self.class2idx={
            'chair':'03001627',
            'table':'04379243',
            'airplan':'02691156',
            'cabinet':'02933112',
            'car':'02958343',
            'watercraft':'04530566',
            'sofa':'04256520',
            'lamp':'03636649'
        }
        self.idx2class = {}
        for key in self.class2idx.keys():
            self.idx2class[self.class2idx[key]] = key
        self.files = []
        self.classes = []
        self.class_idxes = []
        if class_name == 'all':
            for _cn in self.class2idx.keys():
                self.class_idxes.append(self.class2idx[_cn])
        else:
            self.class_idxes.append(self.class2idx[self.class_name])
        for class_idx in self.class_idxes:
            for root, dirs, files in os.walk(os.path.join(path, 'complete', class_idx)):
                files = files 
                break
            for file_name in files:
                self.files.append(os.path.join(class_idx, file_name))
                self.classes.append(self.idx2class[class_idx])
        self.size = len(self.files)
        print_info('Get %d pc of class: %s' % (self.size, self.class_name), prefix=self.prefix)
    
    def __len__(self):
        return self.size 

    def __getitem__(self, index):
        incomplete_pcd_path = os.path.join(self.path, 'partial', self.files[index][:-4], '00.pcd')
        complete_pcd_path = os.path.join(self.path, 'complete', self.files[index])
        incomplete_pcd = read_point_cloud_from_pcd(incomplete_pcd_path)
        complete_pcd = read_point_cloud_from_pcd(complete_pcd_path)

        class_id = self.classes[index]
        complete_pcd, center, max_scale = normalize(complete_pcd, get_arg=True)
        incomplete_pcd = normalize(incomplete_pcd, center=center, max_scale=max_scale)

        # incomplete_pcd = self.clip_points(incomplete_pcd)
        # some real point clouds are not match with gt after clip they contains 0 point
        # we repalce real point clouds with gt point clouds   
        if incomplete_pcd.shape[0] <= 10:
            incomplete_pcd = complete_pcd

        complete_pcd = resample_pcd(complete_pcd, self.gt_pn)
        incomplete_pcd = resample_pcd(incomplete_pcd, self.input_pn)

        return incomplete_pcd, complete_pcd, class_id
    
    @staticmethod
    def to_device(data, device):
        data = list(data)
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        return data

    @staticmethod
    def _collate_fn(datas):
        _incom_points = torch.from_numpy(np.array([data[0] for data in datas]).astype(np.float32))
        _com_points = torch.from_numpy(np.array([data[1] for data in datas]).astype(np.float32))
        _ids = [data[2] for data in datas]
        return _incom_points, _com_points, _ids 


class PCNGANDataset(Dataset):
    def __init__(self, lmdb_realcom_path, lmdb_sn_path, input_pn, gt_pn, class_name='all'):
        super().__init__()
        # print_info('Combine two datasets', prefix='[RealComGAN]')
        self.rc = PcnDataset(lmdb_realcom_path, input_pn, gt_pn, 1, class_name)
        self.sn = ShapeNetDataset(lmdb_sn_path, gt_pn, class_name)
        self.sn_len = len(self.sn)
        self.sn_idx = self.sn_len
        self.sn_index = list(range(self.sn_len))
        np.random.shuffle(self.sn_index)
        self.class_dict = {
            'chair'             :   0,    
            'table'             :   1,    
            'trash_bin'         :   2,    
            'tv_or_monitor'     :   3,        
            'cabinet'           :   4,
            'bookshelf'         :   5,
            'sofa'              :   6,
            'lamp'              :   7,
            'bed'               :   8,    
            'tub'               :   9,    
        }
    
    def get_sn_index(self):
        if self.sn_idx >= self.sn_len:
            self.sn_idx = 0
            np.random.shuffle(self.sn_index)
        # print('sn index', self.sn_idx)
        res = self.sn_index[self.sn_idx]
        # print('shapenet index', res)
        self.sn_idx += 1
        return res        
    
    def __len__(self):
        return len(self.rc)

    def __getitem__(self, index):
        rc_data = self.rc.__getitem__(index)
        sn_data = self.sn.__getitem__(self.get_sn_index())
        return rc_data, sn_data
    
    @staticmethod
    def to_device(data, device):
        rc_data, sn_data = data
        rc_data = PcnDataset.to_device(rc_data, device) 
        sn_data = ShapeNetDataset.to_device(sn_data, device)
        return rc_data, sn_data

    @staticmethod
    def _collate_fn(datas):
        # fix here, datas = [[w, s], [w, s], ...]
        rc_datas = [data[0]for data in datas]
        sn_datas = [data[1]for data in datas]
        rc_datas = PcnDataset._collate_fn(rc_datas)
        sn_datas = ShapeNetDataset._collate_fn(sn_datas)
        return rc_datas, sn_datas