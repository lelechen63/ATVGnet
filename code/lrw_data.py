import matplotlib.backends 
# import face_alignment
import numpy as np
from skimage import io
import pickle
import scipy.io.wavfile
import librosa
import numpy as np
import argparse
import dlib
import cv2
from torch import multiprocessing
import scipy.ndimage as sp
from skimage import transform as tf
import os, fnmatch, shutil
from tqdm import tqdm
import scipy.io.wavfile
from random import shuffle
from scipy.spatial import procrustes
import time
import random
# import matplotlib.animation as manimation
# import matplotlib.lines as mlines
# from matplotlib import transforms
import subprocess
from copy import deepcopy
import python_speech_features
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg')
from mpl_toolkits.mplot3d import Axes3D
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0', flip_input=True)

def parse_arguments():
    """Parse arguments from command line"""
    description = "Train a model."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--root_path', '-p',
                        default="/home/cxu-serve/p1/lchen63/lrw",
                        # default='/media/lele/DATA/lrw/data',
                        help='data path'
                        )
    return parser.parse_args()


args = parse_arguments()
global CC
CC = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../basics/shape_predictor_68_face_landmarks.dat')
def generate_txt():
    txt = open(os.path.join(args.root_path , 'prefix.txt'), 'w')
    for root, dirs, files in os.walk(os.path.join(args.root_path , 'video')):
        for file in files:
            if file[-3:] == 'mp4':
                name = os.path.join(root, file)
                txt.write(name + '\n')


def _extract_images(lists):
    for line in lists:
        temp = line.split('/')
        if not os.path.exists(os.path.join(os.path.join(args.root_path , 'image'),temp[-3])):
            os.mkdir(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]))
        if not os.path.exists(os.path.join(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]), temp[-2])):
            os.mkdir(os.path.join(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]), temp[-2]))


        
        if not os.path.exists(
                os.path.join(os.path.join(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]), temp[-2]), temp[-1][:-4])):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]), temp[-2]), temp[-1][:-4]))

        command = ' ffmpeg -i ' + line + ' -r 25 ' + line.replace('video','image').replace('.mp4','/') +  temp[-1][:-4] + '_%5d.jpg'
        try:
            # pass
            os.system(command)
        except BaseException:
            print (line)
    # else:
    #     continue


def extract_images():
    txt = open(os.path.join(args.root_path , 'prefix.txt'), 'r')
    if not os.path.exists(os.path.join(args.root_path ,  'image')):
        os.mkdir(os.path.join(args.root_path ,  'image'))
    total = []
    for line in txt:
        total.append(line[:-1])
    batch = 1
    datas = []
    batch_size = len(total) / batch
    temp = []
    for i, d in enumerate(total):
        temp.append(d)
        if (i + 1) % batch_size == 0:
            datas.append(temp)
            temp = []

    for i in range(batch):
        process = multiprocessing.Process(
            target=_extract_images, args=(datas[i],))
    process.start()


def _extract_audio(lists):
    for line in lists:
        temp = line.split('/')
        if not os.path.exists(os.path.join(os.path.join(args.root_path , 'audio'),temp[-3])):
            os.mkdir(os.path.join(os.path.join(args.root_path , 'audio'),temp[-3]))
        if not os.path.exists(os.path.join(os.path.join(os.path.join(args.root_path , 'audio'),temp[-3]), temp[-2])):
            os.mkdir(os.path.join(os.path.join(os.path.join(args.root_path , 'audio'),temp[-3]), temp[-2]))

        command = 'ffmpeg -i ' + line + ' -ar 44100  -ac 1  ' + line.replace('video','audio').replace('.mp4','.wav') 
        try:
            # pass
            os.system(command)
        except BaseException:
            print (line)


def extract_audio():
    txt = open(os.path.join(args.root_path , 'prefix.txt'), 'r')
    if not os.path.exists(os.path.join(args.root_path , 'audio')):
        os.mkdir(os.path.join(args.root_path , 'audio'))
    total = []
    for line in txt:
        total.append(line[:-1])
    batch = 1
    datas = []
    batch_size = len(total) / batch
    temp = []
    for i, d in enumerate(total):
        temp.append(d)
        if (i + 1) % batch_size == 0:
            datas.append(temp)
            temp = []
    _extract_audio(datas[0])


def normLmarks(lmarks):
    norm_list = []
    idx = -1
    max_openness = 0.2
    mouthParams = np.zeros((1, 100))
    mouthParams[:, 1] = -0.06
    tmp = deepcopy(MSK)
    tmp[:, 48*2:] += np.dot(mouthParams, SK)[0, :, 48*2:]
    open_mouth_params = np.reshape(np.dot(S, tmp[0, :] - MSK[0, :]), (1, 100))
    if len(lmarks.shape) == 2:
        lmarks = lmarks.reshape(1,68,2)
    for i in range(lmarks.shape[0]):
        mtx1, mtx2, disparity = procrustes(ms_img, lmarks[i, :, :])
        mtx1 = np.reshape(mtx1, [1, 136])
        mtx2 = np.reshape(mtx2, [1, 136])
        norm_list.append(mtx2[0, :])
    pred_seq = []
    init_params = np.reshape(np.dot(S, norm_list[idx] - mtx1[0, :]), (1, 100))
    for i in range(lmarks.shape[0]):
        params = np.reshape(np.dot(S, norm_list[i] - mtx1[0, :]), (1, 100)) - init_params - open_mouth_params
        predicted = np.dot(params, SK)[0, :, :] + MSK
        pred_seq.append(predicted[0, :])
    return np.array(pred_seq), np.array(norm_list), 1


def get_base_mean():

    base_female = np.load('../basics/base_female.npy')
    base_male = np.load('../basics/base_male.npy')

    mean_base = (base_male + base_female ) / 2
    np.save('../basics/base_mean.npy', mean_base)


# get_base_mean()
def get_base(image_path):
    roi,_ = crop_image(image_path, 0, [])
    land2d_p =  '../basics/base.png'
    cv2.imwrite(land2d_p,roi)
    image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(image)[-1]
    np.save('../basics/base_female.npy', preds)
    #TODO: Make this nice
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
    ax.axis('off')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
    ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )
    plt.show()
        
def crop_image(image_path, gg , last):
    image = cv2.imread(image_path)
    if gg == 1:
        [new_y, new_x, r]= last
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        roi = cv2.resize(roi, (256,256), interpolation = cv2.INTER_AREA)

        return roi, last

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = utils.shape_to_np(shape)
      
        (x, y, w, h) = utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(image, "c", (x - 10, y - 10),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # for (x, y) in shape:
        #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        # cv2.putText(image, "a", (x , y ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)
        # cv2.putText(image, "b", (center_x , center_y ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        r = int(0.8 *h)
        new_x = max(center_x - r,0)
        new_y = max(center_y - r,0)

        # cv2.putText(image, "f", (new_x , new_y ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # print (new_x, new_y, r)
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
       
        roi = cv2.resize(roi, (256,256), interpolation = cv2.INTER_AREA)
        # scale =  256. / (2 * r)
       
        # shape = ((shape - np.array([new_x,new_y])) * scale)
        return roi,  [new_y, new_x, r]
    
  

# crop_image('/home/cxu-serve/p1/lchen63/lrw/image/TRADE/test/TRADE_00021/TRADE_00021_019.jpg')
# get_base( '../basics/base_female.png')

def generating_landmark_lips(lists):
    global CC
    image_txt = lists
    name = ''
    last  = []
    
    for line in image_txt:
        try:
            CC = CC + 1

            if CC%100==0:
                print ('{}/{}'.format(CC, len(lists)))
            img_path = line
            temp = img_path.split('/')
            if not os.path.exists(os.path.join(args.root_path,'faces') + '/' + temp[-4]):
                os.mkdir(os.path.join(args.root_path,'faces') + '/' + temp[-4])

            if not os.path.exists(os.path.join(args.root_path,'landmark3d') + '/' + temp[-4]):
                os.mkdir(os.path.join(args.root_path,'landmark3d') + '/' + temp[-4])

            if not os.path.exists(os.path.join(args.root_path,'faces') + '/' + temp[-4] + '/' + temp[-3]):
                os.mkdir(os.path.join(args.root_path,'faces') + '/' + temp[-4] + '/' + temp[-3])

            if not os.path.exists(os.path.join(args.root_path,'landmark3d') + '/' + temp[-4] + '/' + temp[-3]):
                os.mkdir(os.path.join(args.root_path,'landmark3d') + '/' + temp[-4] + '/' + temp[-3])

            if not os.path.exists(
                    os.path.join(args.root_path,'faces') + '/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2]):
                os.mkdir(os.path.join(args.root_path,'faces') + '/' + temp[-4] +
                         '/' + temp[-3] + '/' + temp[-2])

            if not os.path.exists(os.path.join(args.root_path,'landmark3d') + '/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2]):
                os.mkdir(os.path.join(args.root_path,'landmark3d') + '/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2])


            landmark_path = os.path.join(args.root_path,'landmark3d') + '/' + \
                temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.npy'
            lip_path = os.path.join(args.root_path,'faces') + '/' + \
                temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.jpg'
            # if os.path.exists(landmark_path):
            #     continue
            # try:
            if name ==temp[-2]:
                gg = 1
            else:
                gg = 0
            
            roi, last= crop_image(img_path, gg, last)
            cv2.imwrite(lip_path, roi)
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB )
            name = temp[-2]
            preds = fa.get_landmarks(roi)[-1]
            np.save(landmark_path, preds)
            lmark,_,_ = normLmarks( np.load(landmark_path))
            np.save(landmark_path[-4] + '_norm.npy', lmark)
        except:
            print ((img_path, gg, last))
            continue

def __visualization(img_path, fake_lmark):

    temp = img_path.split('/')
    if not os.path.exists( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4])   ):
        os.mkdir( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4]))
    if not os.path.exists( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3] )  ):
        os.mkdir( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3]))
    if not os.path.exists( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3] ,temp[-2] ) ):
        os.mkdir( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3], temp[-2]))

    landmark_path = os.path.join(args.root_path,'landmark3d') + '/' + \
        temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.npy'
    lip_path = os.path.join(args.root_path,'faces') + '/' + \
        temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.jpg'

    img = io.imread(lip_path)
    print (type(img))
    print (img.shape)
    print (img)
    fake_lmark = fake_lmark.view(68,3).numpy()
    preds = np.load(landmark_path)
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img)
    # ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
    ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

    preds = fake_lmark
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='g')
    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='green' )
    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='green')
    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='green')
    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='green')
    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='green')
    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='green')
    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='green')
    ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='green' )

    # ax.view_init(elev=90., azim=90.)
    # ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()
    # print (landmark_path.replace('npy', 'png'))
    # fig.savefig(landmark_path.replace('npy', 'png').replace('landmark3d', 'visualization'))

def __visualization2d(img_path, fake_lmark):

    temp = img_path.split('/')
    if not os.path.exists( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4])   ):
        os.mkdir( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4]))
    if not os.path.exists( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3] )  ):
        os.mkdir( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3]))
    if not os.path.exists( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3] ,temp[-2] ) ):
        os.mkdir( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3], temp[-2]))

    landmark_path = os.path.join('/mnt/ssd0/dat/lchen63/lrw/data','landmark1d') + '/' + \
        temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.npy'
    lip_path = os.path.join(args.root_path,'regions') + '/' + \
        temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.jpg'

    img = io.imread(lip_path)
    print (type(img))
    print (img.shape)
    print (img)
    fake_lmark = fake_lmark.view(68,2).numpy()
    preds = np.load(landmark_path)
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img)
    # ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    # ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], color='blue' )
    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],color='blue')
    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1], color='blue')
    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1], color='blue')
    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1], color='blue')
    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1], color='blue')
    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1], color='blue')
    ax.plot3D(preds[48:,0]*1.2,preds[48:,1], color='blue' )
    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::])

    preds = fake_lmark
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],c="cyan", alpha=1.0, edgecolor='g')
    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], color='green' )
    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1], color='green')
    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1], color='green')
    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1], color='green')
    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1], color='green')
    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1], color='green')
    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1], color='green')
    ax.plot3D(preds[48:,0]*1.2,preds[48:,1], color='green' )

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::])
    plt.show()
    # print (landmark_path.replace('npy', 'png'))
    # fig.savefig(landmark_path.replace('npy', 'png').replace('landmark3d', 'visualization'))

def _visualization(img_path):

    temp = img_path.split('/')
    if not os.path.exists( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4])   ):
        os.mkdir( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4]))
    if not os.path.exists( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3] )  ):
        os.mkdir( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3]))
    if not os.path.exists( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3] ,temp[-2] ) ):
        os.mkdir( os.path.join('/home/cxu-serve/p1/lchen63/lrw/visualization/', temp[-4], temp[-3], temp[-2]))

    landmark_path = os.path.join(args.root_path,'landmark3d') + '/' + \
        temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.npy'
    lip_path = os.path.join(args.root_path,'faces') + '/' + \
        temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.jpg'

    img = io.imread(lip_path)
    preds = np.load(landmark_path)
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2) 
    ax.axis('off')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
    ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    # plt.show()
    print (landmark_path.replace('npy', 'png'))
    fig.savefig(landmark_path.replace('npy', 'png').replace('landmark3d', 'visualization'))



def visualization():
    _file = open(os.path.join('/home/cxu-serve/p1/lchen63/lrw/pickle/', "new_img_full_gt_test.pkl"), "rb")
    data = pickle.load(_file)
    _file.close()

    for index in range(0,len(data),28):
        image_path = data[index][0]
        print (index)
        for ii in range(1,30):
            example_path =   image_path[:-8] + '_%03d.jpg'%ii
            _visualization(example_path)

# visualization()
def multi_pool():
    if not os.path.exists(os.path.join(args.root_path,'faces') ):
        os.mkdir(os.path.join(args.root_path,'faces') )
    if not os.path.exists(os.path.join(args.root_path,'landmark3d') ):
        os.mkdir(os.path.join(args.root_path,'landmark3d') )
    data = []
    words =[]
    for path in os.listdir(os.path.join(args.root_path,'image')):
        words.append(path)
        # break
    words = sorted(words)
    
    # words = [words[0]]
    words = words[int(0.3 *  len(words)):int(0.4 *  len(words))]
    for w in words:

        for path, subdirs, files in os.walk(os.path.join(os.path.join(args.root_path,'image'), w) ):
            for name in files:
                data.append( os.path.join(path, name))
        # if w =='test':
        #     break
    print ('-----')
    print (len(data))
    generating_landmark_lips(data)
    # visualization(data)
    # num_thread = 8
    # datas = []
    # # lock = multiprocessing.Manager().Lock()
    # batch_size = int(len(data) / num_thread)
    # temp = []
    # for i, d in enumerate(data):
    #     temp.append(d)
    #     if (i + 1) % batch_size == 0:
    #         datas.append(temp)
    #         temp = []

    # print ('======')
    # for i in range(num_thread):
        # process = multiprocessing.Process(
        #     target=generating_landmark_lips, args=(datas[i],))
        # process.start()
        
# multi_pool()
def generator_demo_example_lips(img_path):
    print (img_path)
    name = img_path.split('/')[-1]
    landmark_path = os.path.join('../image/', name.replace('jpg', 'npy')) 
    region_path = os.path.join('../image/', name.replace('.jpg', '_region.jpg')) 
    roi, landmark= crop_image(img_path)
    if  np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9:

        # pts2 = np.float32(np.array([template[36],template[45],template[30]]))
        template = np.load( '../basics/base_68.npy')
    else:
        template = np.load( '../basics/base_68_close.npy')
    # pts2 = np.float32(np.vstack((template[27:36,:], template[39,:],template[42,:],template[45,:])))
    pts2 = np.float32(template[27:45,:])
    # pts2 = np.float32(template[17:35,:])
    # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
    pts1 = np.float32(landmark[27:45,:])
    # pts1 = np.float32(landmark[17:35,:])
    tform = tf.SimilarityTransform()
    tform.estimate( pts2, pts1)
    dst = tf.warp(roi, tform, output_shape=(163, 163))

    dst = np.array(dst * 255, dtype=np.uint8)
    dst = dst[1:129,1:129,:]
    cv2.imwrite(region_path, dst)


    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = utils.shape_to_np(shape)
        shape, _ ,_ = normLmarks(shape)
        np.save(landmark_path, shape)
        lmark= shape.reshape(68,2)
        name = region_path.replace('region.jpg','lmark.png')

        utils.plot_flmarks(lmark, name, (-0.2, 0.2), (-0.2, 0.2), 'x', 'y', figsize=(10, 10))

# generator_demo_example_lips('../image/test4.jpg')

# for i in range(4):
#     generator_demo_example_lips('../image/tcd%05d.jpg'%(i + 1))
def generating_demo_landmark_lips(folder = '/mnt/disk1/dat/lchen63/lrw/demo'):
    global CC
    if not os.path.exists(os.path.join(folder,'regions') ):
        os.mkdir(os.path.join(folder,'regions') )
    if not os.path.exists(os.path.join(folder,'landmark1d') ):
        os.mkdir(os.path.join(folder,'landmark1d') )
    this  = 'obama'

    image_txt  = os.listdir(os.path.join(folder,  'image', this))
    image_txt = sorted(image_txt)

    for line in image_txt:
        img_path = os.path.join(folder,  'image', this, line) 
       

        if not os.path.exists(os.path.join(folder,'regions',this) ):
            os.mkdir(os.path.join(folder,'regions',this) )

        if not os.path.exists(os.path.join(folder,'landmark1d',this)):
            os.mkdir(os.path.join(folder,'landmark1d',this) )


        landmark_path =os.path.join(folder,'landmark1d',this, line.replace('jpg','npy'))

        region_path = os.path.join(folder,'regions',this, line)
       
        try:
            print (img_path)
            
            roi, landmark= crop_image(img_path)
            if  np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9:

                # pts2 = np.float32(np.array([template[36],template[45],template[30]]))
                template = np.load( '../basics/base_68.npy')
            else:
                template = np.load( '../basics/base_68_close.npy')
            # pts2 = np.float32(np.vstack((template[27:36,:], template[39,:],template[42,:],template[45,:])))
            pts2 = np.float32(template[27:45,:])
            # pts2 = np.float32(template[17:35,:])
            # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
            pts1 = np.float32(landmark[27:45,:])
            # pts1 = np.float32(landmark[17:35,:])
            tform = tf.SimilarityTransform()
            tform.estimate( pts2, pts1)
            dst = tf.warp(roi, tform, output_shape=(163, 163))

            dst = np.array(dst * 255, dtype=np.uint8)
            dst = dst[1:129,1:129,:]
            cv2.imwrite(region_path, dst)
        

            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            rects = detector(gray, 1)
            for (i, rect) in enumerate(rects):

                shape = predictor(gray, rect)
                shape = utils.shape_to_np(shape)
               
                np.save(landmark_path, shape)
            print (CC )
            CC += 1
            
            
        except:
            cv2.imwrite(region_path, dst)
            np.save(landmark_path, shape)
            continue
            print (line)

# generating_demo_landmark_lips('/mnt/disk1/dat/lchen63/lrw/demo')

def get_data():

    train_set = []
    test_set = []
    count = 0

    for root, dirs, files in os.walk(os.path.join(args.root_path,'landmark1d') ):
        # print count
        # if count == 5:
        #     break
        line = root
        # print line
        # print len(line.split('/'))
        if len(line.split('/'))!= 11:
            continue
        image_path = line
        if '/train/' in image_path:
            train_set.append(image_path)
        elif '/test/' in image_path:
            test_set.append(image_path)
        elif '/val/' in image_path:
            train_set.append(image_path)
        else:
            print ('wrong')
        if len(os.listdir(root)) == 29 * 3 + 1:
            count += 1
    # print len(train_set)
    # print len(test_set)
    # print test_set[0]
    print (count)
    return [train_set, test_set]

def lrw_gt_prepare():
    train_set = []
    test_set = []
    count = 0
    root_path = '/home/cxu-serve/p1/lchen63/lrw/landmark1d/'
    words = os.listdir(root_path)
    words  = sorted(words)
    for word in words:
        print (word)

        # if count >0:
        #     break
        status = ['train', 'val','test']
        for s in status:
            print (count)
            # try:
            for video in  os.listdir( os.path.join(root_path, word, s)):
                total = np.zeros((29,136), dtype=np.float32)
                if len(os.listdir( os.path.join(root_path, word, s, video))) >= 29* 3:
                    for ggg in range(29):
                        current = np.load(os.path.join(root_path, word, s, video, video + '_%03d_norm.npy')%(ggg+1))
                        total[ggg,:] = current
                        
                    np.save( os.path.join(root_path, word, s, video, video + '_norm.npy'), total)
                if os.path.exists( os.path.join(root_path, word, s, video, video + '_norm.npy')):
                    if s == 'train':
                        train_set.append([word, s, video])
                    else:
                        test_set.append([word, s, video])
                    
                else:
                    continue
            # except:
            #     continue
        count += 1
    
    print (len(train_set))
    print (len(test_set))
    datalists =  [train_set, test_set]

    # data = []
    # for dataset in datalists:
    #     temp = []
    #     for i in range(0, len(dataset)):
    #         print ('{}/{}'.format(i, len(dataset)))
    #         lmark_path =os.path.join( dataset[i][0], dataset[i][1],dataset[i][2],dataset[i][2]+'_norm.npy')
    #         if not os.path.exists('/home/cxu-serve/p1/lchen63/lrw/landmark1d/' + lmark_path):
    #                 continue
    #         for  current_frame_id in range(1,29):
    #             img_frame = os.path.join( dataset[i][0], dataset[i][1],dataset[i][2], dataset[i][2]+'_%03d.jpg'%current_frame_id)
                
                
    #             ggg = np.load('/home/cxu-serve/p1/lchen63/lrw/landmark1d/'+ lmark_path)

    #             if ggg.shape != (29,68,2):
    #                 continue
    #             temp.append([img_frame, current_frame_id])
           
    #     data.append(temp)
    
    # print (data[0][0:3])
    # print (len(data[0]))
    # print (len(data[1]))
    # print (len(train_set))
    # print (len(test_set))
    # shuffle(data[0])
    # shuffle(data[1])
    # with open(os.path.join('/home/cxu-serve/p1/lchen63/lrw/','pickle/img_region_gt_warp_train.pkl'), 'wb') as handle:
    #     pickle.dump(data[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join('/home/cxu-serve/p1/lchen63/lrw/','pickle/img_region_gt_warp_test.pkl'), 'wb') as handle:
    #     pickle.dump(data[1], handle, protocol=pickle.HIGHEST_PROTOCOL)


def lrw_lstm_gt_prepare():
    train_set = []
    test_set = []
    count = 0
    root_path = '/mnt/ssd0/dat/lchen63/lrw/data/landmark1d/'
    words =  os.listdir(root_path)
    words  = sorted(words)
    for word in words:
        print(word)
        status =['val','test']
        for s in status:
            try:
                for video in  os.listdir( os.path.join(root_path, word, s)):
                    if os.path.exists( os.path.join(root_path, word, s, video, video + '.npy')):
                        if s == 'train':
                            train_set.append([word, s, video])
                        else:
                            test_set.append([word, s, video])
                        
                    else:
                        continue
            except:
                continue
        count += 1
    print (count)
    print (len(train_set))
    print (len(test_set))
    datalists =  [train_set, test_set]

    data = []
    for dataset in datalists:
        temp = []
        for i in range(0, len(dataset)):
            print ('{}/{}'.format(i, len(dataset)))
            # try:
            lmark_path =os.path.join( dataset[i][0], dataset[i][1],dataset[i][2],dataset[i][2]+'.npy')

            if not os.path.exists(root_path + lmark_path):
                print (lmark_path)
                continue
            else:
                ggg =  np.load(root_path + lmark_path)
                if ggg.shape != (29,68,2) or np.amin(ggg) < 0 or np.amax(ggg) > 127 :
                    continue
                else:
                    for  current_frame_id in range(1,13,4):
                        img_frame = os.path.join( dataset[i][0], dataset[i][1],dataset[i][2], dataset[i][2]+'_%03d.jpg'%current_frame_id)
                        temp.append([img_frame, current_frame_id])

            # except:
            #     continue
        data.append(temp)
    print (data[0][0:3])
    print (len(data[0]))
    print (len(data[1]))
    print (len(train_set))
    print (len(test_set))
    print (data)
    shuffle(data[0])
    shuffle(data[1])

    # with open(os.path.join('/home/cxu-serve/p1/lchen63/lrw/' + 'pickle/region_16_warp_gt_train2.pkl'), 'wb') as handle:
    #     pickle.dump(data[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join('/home/cxu-serve/p1/lchen63/lrw/','pickle/region_16_warp_gt_test2.pkl'), 'wb') as handle:
        pickle.dump(data[1], handle, protocol=pickle.HIGHEST_PROTOCOL)

def audio2mfcc():
    count = 0
   
    root_path = '/mnt/ssd0/dat/lchen63/lrw/data/audio'


    if not os.path.exists('/mnt/ssd0/dat/lchen63/lrw/data/mfcc'):
        os.mkdir('/mnt/ssd0/dat/lchen63/lrw/data/mfcc')

    for word in os.listdir(root_path):
        if not os.path.exists( os.path.join('/mnt/ssd0/dat/lchen63/lrw/data/mfcc', word) ):
            os.mkdir( os.path.join('/mnt/ssd0/dat/lchen63/lrw/data/mfcc', word))


            
        # if count > 50:
        #     break
        status = ['train','val','test']
        for s in status:
            if not os.path.exists( os.path.join('/mnt/ssd0/dat/lchen63/lrw/data/mfcc', word, s) ):
                os.mkdir( os.path.join('/mnt/ssd0/dat/lchen63/lrw/data/mfcc', word, s))
            for name in  os.listdir( os.path.join(root_path, word, s)):

                if name[-3:] == 'wav':
                    filename = os.path.join(root_path, word, s,name)
                    
                    audio,fs = librosa.core.load(filename,sr = 16000 )
                    mfcc = python_speech_features.mfcc(audio,16000,winstep=0.01)

                    mfcc_name = os.path.join('/mnt/ssd0/dat/lchen63/lrw/data/mfcc', word, s, name[:-3] + 'npy')

                    np.save(mfcc_name, mfcc)
        count += 1
        print (count)


def merge_landmark():
    count = 0
    root_path = '/home/cxu-serve/p1/lchen63/lrw/landmark3d'
    for word in os.listdir(root_path):
        status = ['test','train']
        for s in status:
            for video in  os.listdir( os.path.join(root_path, word, s)):
                lmark_path = os.path.join(root_path, word, s, video)
                if len(os.listdir(lmark_path)) < 29 :
                    print (lmark_path)
                    continue 
                else:
                    lmarks = []
                    for i in range(1,30):
                        lmark = np.load( os.path.join(lmark_path, video + '_%03d.npy'%i))
                        lmarks.append(lmark)
                    lmarks = np.array(lmarks)
                    np.save(os.path.join(lmark_path, video + '.npy'), lmarks)
                    print (os.path.join(lmark_path, video + '.npy'))
                    count += 1
                    print (count)

# merge_landmark()


def lrw_demo_prepare():
    train_set = []
    count = 0
    demo_set  = []
    root_path = '/mnt/disk1/dat/lchen63/lrw/demo/regions'
    for word in os.listdir(root_path):
        count += 1
        if count > 50:
            break
        imgs  = sorted(os.listdir(os.path.join(root_path, word)))
        for s in  imgs:
            tmp = []

            tmp.append(os.path.join(word, s))

            lmark_path = os.path.join( word, s.replace('jpg', 'npy'))
            tmp.append(lmark_path)

            demo_set.append(tmp)
    print (len(demo_set))
    print (len(demo_set[0]))
    print (demo_set)
    with open(os.path.join('/mnt/disk1/dat/lchen63/lrw/data','pickle/new_img_full_gt_demo.pkl'), 'wb') as handle:
        pickle.dump(demo_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
def lrw_lstm_demo_prepare():
    train_set = []
    count = 0
    demo_set  = []
    root_path = '/mnt/disk1/dat/lchen63/lrw/demo/regions'
    for word in os.listdir(root_path):
        count += 1
        if count > 50:
            break
        imgs  = sorted(os.listdir(os.path.join(root_path, word)))
        cc = 0
        tmp = []
        for s in  imgs:
            # print (s)
            # print (cc)
            # tmp.append(os.path.join(word, s)

            if cc % 16 == 0:
                
            #lmark_path = os.path.join( word, s.replace('jpg', 'npy'))
            #tmp.append(lmark_path)
                tmp.append(word)
                tmp.append(cc+1)
                demo_set.append(tmp)
                tmp = []
            cc += 1
    print (len(demo_set))
    print (len(demo_set[0]))
    print (demo_set)
    with open(os.path.join('/mnt/disk1/dat/lchen63/lrw/data','pickle/new_16_full_gt_demo.pkl'), 'wb') as handle:
        pickle.dump(demo_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


import torch
def pca_lmark_lrw():
    k = 20
    lmark_root_path = '/mnt/ssd0/dat/lchen63/lrw/data/landmark1d/'
    _file = open( '/home/cxu-serve/p1/lchen63/lrw/pickle/region_16_warp_gt_test2.pkl', "rb")
    train_data = pickle.load(_file)
    shuffle(train_data)
    print (len(train_data))
    landmarks = []
    for index in range(len(train_data)):
        if index == 80000:
            
            break

        lmark_path = os.path.join(lmark_root_path ,train_data[index][0][:-8] + '.npy') 
        t_lmark = np.load(lmark_path)
        landmarks.append(torch.from_numpy(t_lmark[random.randint(0, 28)]))
        print (lmark_path)
    landmarks = torch.stack(landmarks,dim= 0)
    print (landmarks.shape)
    landmarks = landmarks.view(landmarks.size(0), 136)
    mean = torch.mean(landmarks,0)
    print (mean)

    np.save('../basics/mean_lrw_region.npy', mean.numpy())
    landmarks = landmarks - mean.expand_as(landmarks)

    U,S,V  = torch.svd(torch.t(landmarks))
    np.save('../basics/U_lrw_region.npy', U.numpy())
   
def test_pca():
    U = torch.FloatTensor(np.load('../basics/U_lrw_region.npy'))
    mean = torch.FloatTensor(np.load('../basics/mean_lrw_region.npy'))


    _file = open( '/home/cxu-serve/p1/lchen63/lrw/pickle/region_16_warp_gt_train2.pkl', "rb")
    train_data = pickle.load(_file)
    # shuffle(train_data)

    for i in  range(10):
        index = random.randint(0, len(train_data)-1)
        
        lmark_path = os.path.join('/mnt/ssd0/dat/lchen63/lrw/data/landmark1d' ,train_data[index][0][:-8] + '.npy') 
        t_lmark = np.load(lmark_path)

        a = torch.from_numpy(t_lmark).view(29,136)
        for jj in range(29):
            path = os.path.join( '/mnt/ssd0/dat/lchen63/lrw/data/regions/',  train_data[index][0][:-8] + '_%03d.png'%(jj+1) )
            b = a[jj:jj+1]
            d = b - mean.expand_as(b)
            d = torch.mm(d, U[:,:20])



            e = torch.mm( d, U[:,:20].t() )
            e = e + mean.expand_as(e)

            __visualization2d(path, e)






def lrw_lstm_lmark_prepare():
    train_set = []
    test_set = []
    count = 0
    root_path = '/mnt/ssd0/dat/lchen63/lrw/data/landmark1d'
    for word in os.listdir(root_path):
        status = ['test','train']
        for s in status:
            for video in  os.listdir( os.path.join(root_path, word, s)):

                lmark_path = os.path.join(root_path, word, s, video, video + '.npy')

                mfcc_path = os.path.join('/mnt/ssd0/dat/lchen63/lrw/data/mfcc/' , word, s, video + '.npy')

                if os.path.exists(lmark_path) and os.path.exists(mfcc_path):
                    # gg  = np.load(lmark_path)
                    if s == 'train':
                        train_set.append([word, s, video])
                    else:
                        # for xx in range(1,23):
                        test_set.append([word, s, video])
                    count += 1
                    if count % 1000 == 0:

                        print (count)
                else:
                    print ( os.path.join(root_path, word, s, video))
                    continue

    with open(os.path.join('/mnt/ssd0/dat/lchen63/lrw/data','pickle/lmark_train.pkl'), 'wb') as handle:
        pickle.dump(train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join('/mnt/ssd0/dat/lchen63/lrw/data','pickle/lmark_test.pkl'), 'wb') as handle:
        pickle.dump(test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

# pca_seperate_lmark_lrw()

# lrw_lstm_lmark_prepare()
# lrw_lstm_lmark_prepare()
# pca_lmark_lrw()
# test_pca()
lrw_gt_prepare()
# lrw_lstm_gt_prepare()
# pca_lmark_lrw()
# lrw_lstm_demo_prepare()
# lrw_gt_prepare()
# lrw_demo_prepare()
# generate_txt()
# extract_audio()
# extract_images()
# get_base(os.path.join(args.root_path, 'base_roi.jpg'))
# multi_pool()
#merge_landmark()
# multi_pool_lms()
# datalists = get_data()
# print datalists
# generate_image_pickle(datalists)
# generate_video_pickle(datalists)
# generate_video_16_pickle(datalists)
# with open('/home/lele/Music/text-to-image.pytorch/data/train.pkl', 'rb') as handle:
#     b = pickle.load(handle)
# print b[0]

# multi_pool_norm_land()


# def lrw_pca_prepare():
#     train_set = []
#     test_set = []
#     count = 0
#     root_path = '/mnt/disk1/dat/lchen63/lrw/data/regions'
#     for word in os.listdir(root_path):
#         # count += 1
#         # if count > 2:
#         #     break
#         status = ['train','val','test']
#         for s in status:
#             for video in  os.listdir( os.path.join(root_path, word, s)):

#                 if len(os.listdir( os.path.join(root_path, word, s, video))) != 58:
#                     print (os.path.join(root_path, word, s, video))
#                     continue
#                 if s == 'train':
#                     train_set.append([word, s, video])
#                     print (train_set[-1])
#                 else:
#                     test_set.append([word, s, video])
#                     print (test_set[-1])


#     datalists =  [train_set, test_set]

#     data = []
#     for dataset in datalists:
#         temp = []
#         for i in range(0, len(dataset)):
#             print ('{}/{}'.format(i, len(dataset)))
#             for  current_frame_id in range(1,30):
#                 flage = True
#                 img_frame = os.path.join(root_path, dataset[i][0], dataset[i][1],dataset[i][2], dataset[i][2]+'_%03d.jpg'%current_frame_id)
#                 lmark_path =os.path.join('/mnt/disk1/dat/lchen63/lrw/data/landmark_pca', dataset[i][0], dataset[i][1],dataset[i][2]+'.npy')
#                 if np.load(lmark_path).shape != (31,6):
#                     continue
#                 temp.append([img_frame, lmark_path, current_frame_id])
#         data.append(temp)
#     print (data[0][0:3])
#     print (len(data[0]))
#     print (len(data[1]))
#     print (len(train_set))
#     print (len(test_set))
#     with open(os.path.join('/mnt/disk1/dat/lchen63/lrw/data','pickle/new_img_small_train.pkl'), 'wb') as handle:
#         pickle.dump(data[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
#     with open(os.path.join('/mnt/disk1/dat/lchen63/lrw/data','pickle/new_img_small_test.pkl'), 'wb') as handle:
#         pickle.dump(data[1], handle, protocol=pickle.HIGHEST_PROTOCOL)

#'/mnt/disk1/dat/lchen63/lrw/data/landmark1d',
