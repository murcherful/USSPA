import os
import sys
import time

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

from print_util import *


class TestFramework():
    def __init__(self, log_dir, cuda_index, res_dir='res'):
        self.batch_size = 1
        self.log_dir = log_dir
        self.res_dir = os.path.join(log_dir, res_dir)

        if not os.path.exists(self.log_dir):
            self.logger.log_warn('No log dir:' + self.log_dir)
            sys.exit(0)

        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.logger = PrintLogger(self.res_dir, False, prefix='[TEST]')
        if cuda_index is not None and str(self.device) != 'cuda':
            self.logger.log_warn('CUDA device %s is not available' % cuda_index)
        text = 'Device: ' + str(self.device)
        if str(self.device) == 'cuda':
            text += ' ' + cuda_index
        self.logger.log_info(text)
        
    def _set_dataset(self, valid_dataset_name, valid_dataset):
        self.valid_dataset_name = valid_dataset_name
        self.valid_dataset = valid_dataset 
        self.valid_dataset_num = len(self.valid_dataset)
        self.num_work = 4
        self.valid_loader = DataLoader(self.valid_dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.num_work, collate_fn=self.valid_dataset._collate_fn)
        self.valid_step = len(self.valid_loader)
        self.logger.log_info('valid data: ' + self.valid_dataset_name)
        self.logger.log_info('Work num: %d, Valid num: %d, Valid step: %d' % (self.num_work, self.valid_dataset_num, self.valid_step))
        
    def _set_net(self, net, net_name=''):
        self.net = net.to(self.device)
        self.net.eval()
        self.net_name = net_name 
        self.logger.log_info('Net: ' + self.net_name)

    def find_model_path(self, last_epoch):
        if last_epoch is not None:
            last_model_path = os.path.join(self.log_dir, 'model-' + str(last_epoch) + '.pkl') 
            if os.path.exists(last_model_path):
                return last_model_path
            else:
                return None
        else:
            for root, dirs, files in os.walk(self.log_dir):
                break
            epoch = -1
            for f in files:
                s = f.find('-')
                e = f.find('.')
                if s != -1 and e != -1:
                    try:
                        epoch = max(epoch, int(f[s+1:e]))
                    except:
                        continue
            if epoch == -1:
                return None
            else:
                last_model_path = os.path.join(self.log_dir, 'model-%d.pkl' % epoch)
                return last_model_path
    
    '''
    save_func(name, data, outputs, criterion, loss, time)
    '''
    def test(self, save_func, last_epoch=None, save_index=[], debug_func=None, last_model_path=None):
        if last_model_path is None:
            last_model_path = self.find_model_path(last_epoch)
        if last_model_path is None:
            if last_model_path is None:
                self.logger.log_warn('No last model pkl')
            else:
                self.logger.log_warn('No model-%d.pkl' % last_epoch)
            sys.exit(0)
        else:
            self.net.load_state_dict(torch.load(last_model_path))
            self.logger.log_info('Restore last model: ' + last_model_path)
        test_batch_time = 0
        test_num = 0
        test_loss = np.zeros([self.net.loss_test.loss_num])
        loss_dict = {}
        for i, data in enumerate(self.valid_loader):
            start_time = time.time()
            data = self.valid_dataset.to_device(data, self.device)
            info = data[-1]
            # TODO: this part should in Dataset
            #inputs = inputs.to(self.device)#
            #gts = inputs.to(self.device)#
            
            outputs = self.net(data)

            loss = self.net.loss_test(outputs, data)
            
            t = time.time() - start_time
            loss_res = np.array([l.item() for l in loss]) 
            test_loss += loss_res
            test_batch_time += t
            inf = info[0]
            if inf not in loss_dict:
                loss_dict[inf] = [np.zeros([self.net.loss_test.loss_num]), 0]
            loss_dict[inf][0] += loss_res
            loss_dict[inf][1] += 1
            path = os.path.join(self.res_dir, str(i))
            if i in save_index:
                save_func(path, data, outputs, self.net.loss_test, loss, t)
                if debug_func is not None:
                    debug_func(path, self.net)
            log = '<%d>' % i
            for j in range(len(self.net.loss_test.loss_name)):
                log += '%s:%f;' % (self.net.loss_test.loss_name[j], loss[j])
            self.logger.log_info(log)
            
            test_num += 1
        test_batch_time /= test_num
        test_loss /= test_num
        text = ''
        for j, name in enumerate(self.net.loss_test.loss_name):
            text += '%s: %f, ' % (name, test_loss[j])
        self.logger.log_info(text)
        avg = None
        avg_num = 0
        class_avg = None
        class_avg_num = 0
        for obj_class in loss_dict:
            text = '%7s: ' % obj_class[:7]
            obj_loss = loss_dict[obj_class][0]
            obj_num = loss_dict[obj_class][1]
            if avg_num == 0:
                avg = obj_loss
                avg_num = obj_num
                class_avg = obj_loss/obj_num
                class_avg_num += 1
            else:
                avg += obj_loss 
                avg_num += obj_num
                class_avg += obj_loss/obj_num 
                class_avg_num += 1
            for l in obj_loss:
                text += '%.7f, ' % (l/obj_num)
            self.logger.log_info(text)
        if avg_num != 0:
            text = '%7s: ' % 'AVG'
            for l in avg:
                text += '%.7f, ' % (l/avg_num)
            self.logger.log_info(text)
        if class_avg_num != 0:
            text = '%7s: ' % 'CLASSAVG'
            for l in class_avg:
                text += '%.7f, ' % (l/class_avg_num)
            self.logger.log_info(text)
        self.logger.log_info('test batch time: %f' % (test_batch_time))



