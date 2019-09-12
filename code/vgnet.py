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
from dataset import  LRWdataset1D_lstm_gt 
from models import VG_net 
from models import GL_Discriminator

from torch.nn import init
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

        self.generator = VG_net()
        self.discriminator = GL_Discriminator()
        self.bce_loss_fn = nn.BCELoss()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config
        self.ones = Variable(torch.ones(config.batch_size), requires_grad=False)
        self.zeros = Variable(torch.zeros(config.batch_size), requires_grad=False)

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            # self.encoder = nn.DataParallel(self.encoder.cuda(device=config.cuda1), device_ids=device_ids)
            self.generator     = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            self.discriminator     = nn.DataParallel(self.discriminator, device_ids=device_ids).cuda()

            self.bce_loss_fn   = self.bce_loss_fn.cuda(device=config.cuda1)
            self.mse_loss_fn   = self.mse_loss_fn.cuda(device=config.cuda1)
            self.l1_loss_fn = self.l1_loss_fn.cuda(device=config.cuda1)
            self.ones          = self.ones.cuda(device=config.cuda1)
            self.zeros          = self.zeros.cuda(device=config.cuda1)
# #########single GPU#######################

#         if config.cuda:
#             device_ids = [int(i) for i in config.device_ids.split(',')]
#             self.generator     = self.generator.cuda(device=config.cuda1)
#             self.encoder = self.encoder.cuda(device=config.cuda1)
#             self.mse_loss_fn   = self.mse_loss_fn.cuda(device=config.cuda1)
#             self.l1_loss_fn =  nn.L1Loss().cuda(device=config.cuda1)
        initialize_weights(self.generator)
        self.start_epoch = 0
        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)
        print ('-----------')
       

        self.opt_g = torch.optim.Adam( self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.opt_d = torch.optim.Adam( self.discriminator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.dataset = LRWdataset1D_lstm_gt(config.dataset_dir, train=config.is_train)
        
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        data_iter = iter(self.data_loader)
        data_iter.next()

    def fit(self):

        config = self.config

        musk = torch.FloatTensor(1,136)
        musk[:,:] = 1.0
        for jj in range(0,40,2):
            musk[:,(97+jj) ] = 100.0
        musk = musk.unsqueeze(0)

        num_steps_per_epoch = len(self.data_loader)
        cc = 0
        t0 = time.time()
        xLim=(-1.0, 1.0)
        yLim=(-1.0,1.0)
        xLab = 'x'
        yLab = 'y'

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, (example_img, example_landmark, right_img,right_landmark) in enumerate(self.data_loader):
                t1 = time.time()

                if config.cuda:
                    example_img = Variable(example_img.float()).cuda(device=config.cuda1)
                    example_landmark = Variable(example_landmark.float()).cuda(device=config.cuda1)
                    right_img    = Variable(right_img.float()).cuda(device=config.cuda1)
                    right_landmark = Variable(right_landmark.float()).cuda(device=config.cuda1)
                    musk = Variable(musk.float()).cuda(device=config.cuda1)
                else:
                    example_img = Variable(example_img.float())

                    right_img = Variable(right_img.float())
                    mfccs = Variable(mfccs.float())
                #train the discriminator
                for p in self.discriminator.parameters():
                    p.requires_grad =  True

                fake_im, _ ,_,_ = self.generator(example_img, right_landmark, example_landmark)
                D_real, d_r_lmark = self.discriminator(right_img, example_landmark)
                loss_real = self.bce_loss_fn(D_real, self.ones)

                loss_lmark_r =  self.mse_loss_fn(d_r_lmark * musk, right_landmark* musk)
                # train with fake image
                D_fake, d_f_lmark = self.discriminator(fake_im.detach(), example_landmark)
                loss_fake = self.bce_loss_fn(D_fake, self.zeros)
                loss_lmark_f =  self.mse_loss_fn(d_f_lmark * musk, right_landmark* musk)

                loss_disc = loss_real  + loss_fake + loss_lmark_f + loss_lmark_r

                loss_disc.backward()
                self.opt_d.step()
                self._reset_gradients()

                #train the generator
                for p in self.discriminator.parameters():
                    p.requires_grad = False  # to avoid computation

                fake_im, att ,colors, _ = self.generator(example_img,  right_landmark, example_landmark)
                D_fake, d_f_lmark = self.discriminator(fake_im, example_landmark)

                loss_lmark =  self.mse_loss_fn(d_f_lmark * musk, right_landmark* musk)
                loss_gen = self.bce_loss_fn(D_fake, self.ones)

                loss_musk = att.detach() + 0.5
                diff = torch.abs(fake_im - right_img) * loss_musk
                loss_pix = torch.mean(diff)
                loss =10 * loss_pix + loss_gen + loss_lmark
                loss.backward() 
                self.opt_g.step()
                self._reset_gradients()
                t2 = time.time()

                if (step+1) % 10 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch

                    print("[{}/{}][{}/{}]  ,  loss_disc: {:.8f},   loss_lmark_f: {:.8f},  loss_lmark_r: {:.8f} ,  loss_gen: {:.8f}  ,  loss_pix: {:.8f} ,  loss_lmark: {:.8f} , data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss_disc.data[0],loss_lmark_f.data[0],loss_lmark_r.data[0],loss_gen.data[0],loss_pix.data[0],loss_lmark.data[0], t1-t0,  t2 - t1))

                if (step) % (int(num_steps_per_epoch  / 20 )) == 0 :
                    atts_store = att.data.contiguous().view(config.batch_size*16,1,128,128)
                    colors_store = colors.data.contiguous().view(config.batch_size * 16,3,128,128)
                    fake_store = fake_im.data.contiguous().view(config.batch_size*16,3,128,128)
                    torchvision.utils.save_image(atts_store, 
                        "{}att_{}.png".format(config.sample_dir,cc),normalize=True)
                    torchvision.utils.save_image(colors_store, 
                        "{}color_{}.png".format(config.sample_dir,cc),normalize=True)
                    torchvision.utils.save_image(fake_store,
                        "{}imf_fake_{}.png".format(config.sample_dir,cc),normalize=True)
                    real_store = right_img.data.contiguous().view(config.batch_size * 16,3,128,128)
                    torchvision.utils.save_image(real_store,
                        "{}img_real_{}.png".format(config.sample_dir,cc),normalize=True)
                    cc += 1
                    torch.save(self.generator.state_dict(),
                               "{}/vg_net_{}.pth"
                               .format(config.model_dir,cc))
                 
                t0 = time.time()
    def load(self, directory, epoch):
        gen_path = os.path.join(directory, 'generator_{}.pth'.format(epoch))

        self.generator.load_state_dict(torch.load(gen_path))

        dis_path = os.path.join(directory, 'discriminator_{}.pth'.format(epoch))

        self.discriminator.load_state_dict(torch.load(dis_path))


    def _reset_gradients(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

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
                        default=10)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=5)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/mnt/ssd0/dat/lchen63/lrw/data/pickle/")
                        # default="/mnt/ssd0/dat/lchen63/grid/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument("--model_dir",
                        type=str,
                        default="/u/lchen63/lrw/sample/demo")
                        # default="/mnt/disk1/dat/lchen63/grid/model/model_gan_r")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="/u/lchen63/lrw/sample/demo/")
                        # default="/mnt/disk1/dat/lchen63/grid/sample/model_gan_r/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')

    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='lrw')
    parser.add_argument('--num_thread', type=int, default=4)
    # parser.add_argument('--flownet_pth', type=str, help='path of flownets model')
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--perceptual', type=bool, default=False)

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":
    config = parse_args()
    config.is_train = 'train'
    import vgnet as trainer
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    config.cuda1 = torch.device('cuda:{}'.format(config.device_ids))
    main(config)

