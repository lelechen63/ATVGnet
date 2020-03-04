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

from torch.nn import init
import utils
def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

class Trainer():
    def __init__(self, config):
        if config.lstm == True:
            self.generator = AT_net()
        else:
            self.generator = AT_single()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            self.generator     = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            # self.generator     = self.generator.cuda()
            self.mse_loss_fn   = self.mse_loss_fn.cuda()
            self.l1_loss_fn = self.l1_loss_fn.cuda()
# #########single GPU#######################

#         if config.cuda:
#             device_ids = [int(i) for i in config.device_ids.split(',')]
#             self.generator     = self.generator.cuda()
#             self.encoder = self.encoder.cuda()
#             self.mse_loss_fn   = self.mse_loss_fn.cuda()
#             self.l1_loss_fn =  nn.L1Loss().cuda()
        initialize_weights(self.generator)
        self.start_epoch = 0
        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)
        self.opt_g = torch.optim.Adam( self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        if config.lstm:
            if config.dataset == 'lrw':
                self.dataset = LRW_1D_lstm_landmark_pca(config.dataset_dir, train=config.is_train)
            else:
                self.dataset = GRID_1D_lstm_landmark_pca(config.dataset_dir, train=config.is_train)
        else:
            if config.dataset == 'lrw':
                self.dataset = LRW_1D_single_landmark_pca(config.dataset_dir, train=config.is_train)
            else:
                self.dataset = GRID_1D_single_landmark_pca(config.dataset_dir, train=config.is_train)



        # elif config.dataset == 'ldc':
        #     self.dataset = LDCDataset(config.dataset_dir, train=config.is_train)

        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        

    def fit(self):
        config = self.config
        if config.dataset == 'grid':

            pca = torch.FloatTensor( np.load('../basics/U_grid.npy')[:,:6]).cuda()
            mean =torch.FloatTensor( np.load('../basics/mean_grid.npy')).cuda()
        elif config.dataset == 'lrw':

            pca = torch.FloatTensor(np.load('../basics/U_lrw1.npy')[:,:6] ).cuda()
            mean = torch.FloatTensor(np.load('../basics/mean_lrw1.npy')).cuda()
        else:
            raise Exception('wrong key word for the dataset input')

        num_steps_per_epoch = len(self.data_loader)
        cc = 0
        t0 = time.time()
        xLim=(-1.0, 1.0)
        yLim=(-1.0, 1.0)
        xLab = 'x'
        yLab = 'y'
        

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, (example_landmark, example_audio, lmark, audio) in enumerate(self.data_loader):
                t1 = time.time()

                if config.cuda:
                    lmark    = Variable(lmark.float()).cuda()
                    audio = Variable(audio.float()).cuda()
                    example_landmark = Variable(example_landmark.float()).cuda()
                else:
                    lmark    = Variable(lmark.float())
                    audio = Variable(audio.float())
                    example_landmark = Variable(example_landmark.float())

                fake_lmark= self.generator( example_landmark, audio)

                loss =  self.mse_loss_fn(fake_lmark , lmark) 
                loss.backward() 
                self.opt_g.step()
                self._reset_gradients()


                if (step+1) % 10 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch

                    print("[{}/{}][{}/{}]   loss1: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  t1-t0,  time.time() - t1))
                if (step) % (int(num_steps_per_epoch  / 10 )) == 0 and step != 0:
                    if config.lstm:
                        lmark = lmark.view(lmark.size(0) * lmark.size(1) , 6)
                        lmark = torch.mm( lmark, pca.t() ) 
                        lmark = lmark + mean.expand_as(lmark)
                        fake_lmark = fake_lmark.view(fake_lmark.size(0) * fake_lmark.size(1) , 6) 
                        fake_lmark[:, 1:6] *= 2*torch.FloatTensor(np.array([1.1, 1.2, 1.3, 1.4, 1.5])).cuda() 
                        fake_lmark = torch.mm( fake_lmark, pca.t() ) 
                        fake_lmark = fake_lmark + mean.expand_as(fake_lmark)
                        lmark = lmark.view(config.batch_size, 16, 136)
                        lmark = lmark.data.cpu().numpy()
                        fake_lmark = fake_lmark.view(config.batch_size, 16, 136)
                        fake_lmark = fake_lmark.data.cpu().numpy()
                        for indx in range(3):
                            for jj in range(16):

                                name = "{}real_{}_{}_{}.png".format(config.sample_dir,cc, indx,jj)
                                utils.plot_flmarks(lmark[indx,jj], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
                                name = "{}fake_{}_{}_{}.png".format(config.sample_dir,cc, indx,jj)
                                utils.plot_flmarks(fake_lmark[indx,jj], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
                        torch.save(self.generator.state_dict(),
                                   "{}/atnet_lstm_{}.pth"
                                   .format(config.model_dir,cc))
                    else:
                        lmark = torch.mm( lmark, pca.t() ) 
                        lmark = lmark + mean.expand_as(lmark)
                        fake_lmark[ 1:6] *= 2*torch.FloatTensor(np.array([1.1, 1.2, 1.3, 1.4, 1.5])).cuda() 
                        fake_lmark = torch.mm( fake_lmark, pca.t() )  
                        fake_lmark = fake_lmark + mean.expand_as(fake_lmark)
                        lmark = lmark.data.cpu().numpy()
                        fake_lmark = fake_lmark.data.cpu().numpy()
                        for indx in range(0,config.batch_size, 8):
                            name = "{}single_real_{}_{}.png".format(config.sample_dir,cc, indx)
                            utils.plot_flmarks(lmark[indx], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
                            name = "{}single_fake_{}_{}.png".format(config.sample_dir,cc, indx)
                            utils.plot_flmarks(fake_lmark[indx], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
                            
                        torch.save(self.generator.state_dict(),
                                   "{}anet_single_{}.pth"
                                   .format(config.model_dir,cc))
                    cc += 1
                 
                t0 = time.time()
 
    def _reset_gradients(self):
        self.generator.zero_grad()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=10)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="../dataset/")
                        # default="/mnt/ssd0/dat/lchen63/grid/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument("--model_dir",
                        type=str,
                        default="../model/atnet/")
                        # default="/mnt/disk1/dat/lchen63/grid/model/model_gan_r")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="../sample/atnet/")
                        # default="/mnt/disk1/dat/lchen63/grid/sample/model_gan_r/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='lrw')
    parser.add_argument('--lstm', type=bool, default= True)
    parser.add_argument('--num_thread', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--rnn', type=bool, default=True)

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":

    config = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.is_train = 'train'
    import atnet as trainer
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    config.cuda1 = torch.device('cuda:{}'.format(config.device_ids))
    main(config)

