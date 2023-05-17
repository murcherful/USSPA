import os
import sys 

# cuda_index = '1'
cuda_index = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('../util')
from my_dataset import PCNGANDataset
from train_util_gan import TrainFramework

import argparse

from usspa_pcn import USSPA


def train(args):
    train_dataset = PCNGANDataset(args.lmdb_train, args.lmdb_sn, args.input_pn, args.gt_pn, args.class_name)
    valid_dataset = PCNGANDataset(args.lmdb_valid, args.lmdb_sn, args.input_pn, args.gt_pn, args.class_name)

    net = USSPA()

    tf = TrainFramework(args.batch_size, args.log_dir, args.restore, cuda_index)
    tf._set_dataset(args.lmdb_train, args.lmdb_valid, train_dataset, valid_dataset)
    tf._set_net(net, 'USSPA_PCN')
    tf._set_optimzer(args.opt, lr=args.lr, weight_decay=args.weight_decay)
    tf.train(args.max_epoch, G_opt_step=1, D_opt_step=1, save_pre_epoch=10, print_pre_step=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lmdb_train', default='/home/mcf/data/works/realcom/data/PCN/train')
    parser.add_argument('--lmdb_valid', default='/home/mcf/data/works/realcom/data/PCN/test')
    parser.add_argument('--lmdb_sn', default='/home/mcf/data/works/realcom/data/RealComShapeNetData/shapenet_data.lmdb')

    parser.add_argument('--class_name', default='chair', choices=['chair', 'table', 'cabinet', 'sofa', 'lamp'])

    parser.add_argument('--restore', action='store_true')
    
    parser.add_argument('--log_dir', default='log_pcn_') 

    parser.add_argument('--opt', default='Adam')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--input_pn', type=int, default=2048)
    parser.add_argument('--gt_pn', type=int, default=2048)
    parser.add_argument('--max_epoch', type=int, default=480)
    
    args = parser.parse_args()
    args.log_dir = args.log_dir + args.class_name
    train(args)