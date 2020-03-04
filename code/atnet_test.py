
import matplotlib 
import matplotlib.pyplot as plt
import argparse
import scipy.misc
import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import numpy as np
from collections import OrderedDict
import argparse

from dataset import  GRID_1D_lstm_landmark_pca, LRW_1D_lstm_landmark_pca, GRID_1D_single_landmark_pca, LRW_1D_single_landmark_pca

from models import AT_single, AT_net

import utils

def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def parse_args():
    parser = argparse.ArgumentParser()
  
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="../dataset/")
                        # default="/mnt/disk1/dat/lchen63/lrw/data/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument("--model_name",
                        type=str,
                        default="../model/atnet/audio2lmark_24.pth")
                        # default="/mnt/disk1/dat/lchen63/lrw/model/model_gan_r2/r_generator_38.pth")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="../sample/atnet_test/")
                        # default="/mnt/disk1/dat/lchen63/lrw/test_result/model_gan_r2/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='lrw')
    parser.add_argument('--lstm', type=bool, default=True)
    parser.add_argument('--num_thread', type=int, default=1)
    # parser.add_argument('--flownet_pth', type=str, help='path of flownets model')
   

    return parser.parse_args()
config = parse_args()


def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.is_train = 'test'
    if config.lstm == True:
        generator = AT_net()
    else:
        generator = AT_single()
    if config.dataset == 'grid':

        pca = torch.FloatTensor( np.load('../basics/U_grid.npy')[:,:6]).cuda()
        mean =torch.FloatTensor( np.load('../basics/mean_grid.npy')).cuda()
    elif config.dataset == 'lrw':

        pca = torch.FloatTensor(np.load('../basics/U_lrw1.npy')[:,:6] ).cuda()
        mean = torch.FloatTensor(np.load('../basics/mean_lrw1.npy')).cuda()
    else:
        raise Exception('wrong key word for the dataset input')
    xLim=(-1.0, 1.0)
    yLim=(-1.0,1.0)
    xLab = 'x'
    yLab = 'y'

    state_dict = multi2single(config.model_name, 1)
    generator.load_state_dict(state_dict)
    print ('load pretrained [{}]'.format(config.model_name))

    if config.lstm:
        if config.dataset == 'lrw':
            dataset = LRW_1D_lstm_landmark_pca(config.dataset_dir, train=config.is_train)
        else:
            dataset = GRID_1D_lstm_landmark_pca(config.dataset_dir, train=config.is_train)
    else:
        if config.dataset == 'lrw':
            dataset = LRW_1D_single_landmark_pca(config.dataset_dir, train=config.is_train)
        else:
            dataset = GRID_1D_single_landmark_pca(config.dataset_dir, train=config.is_train)
    data_loader = DataLoader(dataset,
                                      batch_size=config.batch_size,
                                      num_workers= config.num_thread,
                                      shuffle=False, drop_last=True)
    data_iter = iter(data_loader)
    data_iter.next()
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    if not os.path.exists(os.path.join(config.sample_dir, 'fake')):
        os.mkdir(os.path.join(config.sample_dir, 'fake'))
    if not os.path.exists(os.path.join(config.sample_dir, 'real')):
        os.mkdir(os.path.join(config.sample_dir, 'real'))
    if config.cuda:
        generator = generator.cuda()
    generator.eval()
    for step,  (example_landmark, example_audio, lmark, audio) in enumerate(data_loader):
        with torch.no_grad():
            print (step)
            if step == 5:
                break
            if config.cuda:
                example_audio    = Variable(example_audio.float()).cuda()
                lmark    = Variable(lmark.float()).cuda()
                audio = Variable(audio.float()).cuda()
                example_landmark = Variable(example_landmark.float()).cuda() 
            if config.lstm:
                fake_lmark= generator(example_landmark, audio)
                fake_lmark = fake_lmark.view(fake_lmark.size(0) * fake_lmark.size(1) , 6) 
                fake_lmark[:, 1:6] *= 2*torch.FloatTensor(np.array([1.2, 1.4, 1.6, 1.8, 2.0])).cuda() 
                fake_lmark = torch.mm( fake_lmark, pca.t() )
                fake_lmark = fake_lmark + mean.expand_as(fake_lmark)
                fake_lmark = fake_lmark.view(config.batch_size, 16, 136)
                fake_lmark = fake_lmark.data.cpu().numpy() 
                lmark = lmark.view(lmark.size(0) * lmark.size(1) , 6)
                lmark = torch.mm( lmark, pca.t() ) 
                lmark = lmark + mean.expand_as(lmark)
                lmark = lmark.view(config.batch_size, 16, 136)
                lmark = lmark.data.cpu().numpy()

                for indx in range(config.batch_size):
                    for jj in range(16):
                        name = "{}real_{}_{}_{}.png".format(config.sample_dir,step, indx,jj)
                        utils.plot_flmarks(lmark[indx,jj], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
                        name = "{}fake_{}_{}_{}.png".format(config.sample_dir,step, indx,jj)
                        utils.plot_flmarks(fake_lmark[indx,jj], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
            else:
                fake_lmark= generator(example_landmark, audio)
                fake_lmark[:, 1:6] *= 2*torch.FloatTensor(np.array([1.2, 1.4, 1.6, 1.8, 2.0])).cuda() 
                fake_lmark = torch.mm( fake_lmark, pca.t() )
                fake_lmark = fake_lmark + mean.expand_as(fake_lmark)
                fake_lmark = fake_lmark.data.cpu().numpy()
                lmark = torch.mm( lmark, pca.t() )
                lmark = lmark + mean.expand_as(lmark)
                lmark = lmark.data.cpu().numpy()

                for indx in range(config.batch_size):
                    name = '{}real/real_{}.png'.format(config.sample_dir,step *config.batch_size + indx)

                    utils.plot_flmarks(lmark[indx], name, xLim, yLim, xLab, yLab, figsize=(10, 10))

                    name  = '{}fake/fake_{}.png'.format(config.sample_dir,step *config.batch_size + indx)

                    utils.plot_flmarks(fake_lmark[indx], name, xLim, yLim, xLab, yLab, figsize=(10, 10))


test()