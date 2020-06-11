
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
from dataset import LRWdataset1D_lstm_gt as LRWdataset
from models import VG_net



def multi2single(model_path, id):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
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
                        default="/mnt/ssd0/dat/lchen63/lrw/data/pickle/")
                        # default="/mnt/ssd0/dat/lchen63/grid/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument("--model_name",
                        type=str,
                        default="/u/lchen63//lrw/sample/demo/vg_net_5.pth")
                        # default="/mnt/disk1/dat/lchen63/lrw/model/model_gan_r2/r_generator_38.pth")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="/u/lchen63/grid/test_result/gan_lstm/")
                        # default="/mnt/disk1/dat/lchen63/lrw/test_result/model_gan_r2/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')
   
   
    parser.add_argument('--device_ids', type=str, default='1')
    parser.add_argument('--dataset', type=str, default='lrw')
    parser.add_argument('--num_thread', type=int, default=1)
    # parser.add_argument('--flownet_pth', type=str, help='path of flownets model')
   

    return parser.parse_args()
config = parse_args()

def bilinear_interpolate_torch_gridsample(image):
	# image = image.permute(0,2,1,3,4)
	print (image.size())
	image = image.view(image.size(0) * image.size(1),image.size(2),image.size(3),image.size(4))
	new_img =  torch.nn.functional.upsample(image, size=(image.size(3)*4, image.size(3)*4), mode='bilinear')  
	print (new_img.size())
	new_img = new_img.view(config.batch_size,1,16,image.size(3)*4, image.size(3)*4)
	return new_img


def test():
	os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
	generator = VG_net()
	if config.cuda:
	    generator = generator.cuda()

	state_dict = multi2single(config.model_name, 1)
	generator.load_state_dict(state_dict)
	print ('load pretrained [{}]'.format(config.model_name))
	generator.eval()
	dataset =  LRWdataset(
	            config.dataset_dir, train='test')
	data_loader = DataLoader(dataset,
	                                  batch_size=config.batch_size,
	                                  num_workers= config.num_thread,
	                                  shuffle=True, drop_last=True)
	data_iter = iter(data_loader)
	data_iter.next()


	if not os.path.exists(config.sample_dir):
		os.mkdir(config.sample_dir)
	if not os.path.exists(os.path.join(config.sample_dir, 'fake')):
		os.mkdir(os.path.join(config.sample_dir, 'fake'))
	if not os.path.exists(os.path.join(config.sample_dir, 'real')):
		os.mkdir(os.path.join(config.sample_dir, 'real'))
	if not os.path.exists(os.path.join(config.sample_dir, 'color')):
		os.mkdir(os.path.join(config.sample_dir, 'color'))
	if not os.path.exists(os.path.join(config.sample_dir, 'single')):
		os.mkdir(os.path.join(config.sample_dir, 'single'))
	generator.eval()
	for step, (example_img, example_landmark, real_im, landmarks) in enumerate(data_loader):
		with torch.no_grad():
			if step == 43:
				break
			if config.cuda:
			    real_im = Variable(real_im.float()).cuda()
			    example_img = Variable(example_img.float()).cuda() 
			    landmarks = Variable(landmarks.float()).cuda()
			    example_landmark = Variable(example_landmark.float()).cuda()
			fake_im, atts, colors, lmark_att = generator(example_img,  landmarks, example_landmark)

			fake_store = fake_im.data.contiguous().view(config.batch_size*16,3,128,128)
			real_store = real_im.data.contiguous().view(config.batch_size * 16,3,128,128)
			atts_store = atts.data.contiguous().view(config.batch_size*16,1,128,128)
			colors_store = colors.data.contiguous().view(config.batch_size * 16,3,128,128)
			lmark_att = bilinear_interpolate_torch_gridsample(lmark_att)
			lmark_att_store = lmark_att.data.contiguous().view(config.batch_size*16,1,128,128)
			torchvision.utils.save_image(atts_store, 
			   "{}color/att_{}.png".format(config.sample_dir,step),normalize=False, range=[0,1])
			torchvision.utils.save_image(lmark_att_store, 
			   "{}color/lmark_att_{}.png".format(config.sample_dir,step),normalize=False, range=[0,1])
			torchvision.utils.save_image(colors_store, 
			   "{}color/color_{}.png".format(config.sample_dir,step),normalize=True)
			torchvision.utils.save_image(fake_store, 
			    "{}fake/fake_{}.png".format(config.sample_dir,step),normalize=True)

			torchvision.utils.save_image(real_store,
			    "{}real/real_{}.png".format(config.sample_dir,step),normalize=True)
				
			fake_store = fake_im.data.cpu().permute(0, 1,3,4, 2).view(config.batch_size*16,128,128,3).numpy()
			real_store = real_im.data.cpu().permute(0, 1,3,4,2).view(config.batch_size*16,128,128,3).numpy()
			
			print (step)
			print (fake_store.shape)
			for inx in range(config.batch_size *16 ):
				    scipy.misc.imsave("{}single/fake_{}.png".format(config.sample_dir,step*config.batch_size * 16+inx),fake_store[inx])
				    scipy.misc.imsave("{}single/real_{}.png".format(config.sample_dir,step*config.batch_size * 16+inx),real_store[inx])
			   

test()