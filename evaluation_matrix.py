import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import glob
import torch
import scipy.misc
import numpy 
import argparse
import cv2
from scipy import signal
from scipy import ndimage
from collections import OrderedDict
import multiprocessing
import math
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
from skimage.measure import compare_ssim as ssim_f
from PIL import Image
import numpy as np
from skimage import feature
import math
from scipy.ndimage import correlate
import imutils
import dlib
from imutils import face_utils
import csv

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--sample_dir",
                        type=str,
                        default="/u/lchen63/lrw/test_result")
                        # default="/u/lchen63/grid/test_result")
    parser.add_argument("--model_name",
                        type=str,
                        default="musk_r_lstm_newloss_row")
    parser.add_argument("--pickle_path",
                        type = str,
                        default ='/u/lchen63/lrw/test_pickle/')
    parser.add_argument("--num_thread",
                        type=int,
                        default=5)
    return parser.parse_args()


def get_pickle(config):
    sample_dir = config.sample_dir
    test_data = []
    print ('------------------------')
    for i in range(1000):
        tmp = {}
        tmp["real_path"] = []
        tmp["fake_path"] = []
        
        for j in range(8):
            real_path = os.path.join(sample_dir, 'single', 'real_%d.png'%(i*8+j))
            fake_path = os.path.join(sample_dir, 'single', 'fake_%d.png'%(i*8+j)) 
            if not os.path.exists(real_path):
                print (real_path)
            
            elif not os.path.exists(fake_path):
                print (fake_path)
            else:
                tmp["real_path"].append(real_path)
                tmp["fake_path"].append(fake_path)
        test_data.append(tmp)
    with open(os.path.join(config.pickle_path,'{}.pkl'.format(config.model_name)), 'wb') as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_pickle_xface(csv_file, config):
    _csv = open(csv_file,'rb')
    reader = csv.reader(_csv)

    test_data = []
    for line in reader:
        fid = line[2]
        fimg_path = line[1].replace('.wav','_' + fid + '.jpg')
        print (fimg_path)
        rimg_path = line[0]
        print (rimg_path)
        tmp = {}
        tmp["real_path"] = [rimg_path]
        tmp["fake_path"] = [fimg_path]
        test_data.append(tmp)
    with open(os.path.join(config.pickle_path,'{}.pkl'.format(config.model_name)), 'wb') as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

##########################################################
def get_pickle_yousaidthat(csv_file, config):
    _csv = open(csv_file,'rb')
    reader = csv.reader(_csv)

    test_data = []
    for line in reader:
        fid = line[6]
        for  i in range(1,30):
            fimg_path = fid + 'fake_%03d.jpg'%i
            rimg_path = fid + '%03d.jpg'%i
            if not os.path.exists(fimg_path):
                print (fimg_path)
                continue
            elif not os.path.exists(rimg_path):
                print (rimg_path)
                continue
            else:

                tmp = {}
                tmp["real_path"] = [rimg_path]
                tmp["fake_path"] = [fimg_path]
                test_data.append(tmp)
    with open(os.path.join(config.pickle_path,'{}.pkl'.format(config.model_name)), 'wb') as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


##################CPDB##############

def block_process(A, block):
    block_contrast = np.zeros((A.shape[0]/block[0], A.shape[1]/block[1]), dtype=np.int32)
    flatten_contrast = list()
    for i in range(0, A.shape[0], block[0]):
        for j in range(0, A.shape[1], block[1]):
            block_view = A[i:i+block[0], j:j+block[1]]
            block_view = np.max(block_view) - np.min(block_view)
            flatten_contrast.append(block_view)
    block_contrast = np.array(flatten_contrast).reshape(block_contrast.shape)
    return block_contrast
def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def ssim_ff(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

def cpbd_compute(image):
    if isinstance(image, str):
        image = Image.open(image)
        image = image.convert('L')
    img = np.array(image, dtype=np.float32)
    m, n = img.shape

    threshold = 0.002
    beta = 3.6
    rb = 64
    rc = 64
    max_blk_row_idx = int(m/rb)
    max_blk_col_idx = int(n/rc)
    widthjnb = np.array([np.append(5 * np.ones((1, 51)), 3*np.ones((1, 205)))])
    total_num_edges = 0
    hist_pblur = np.zeros(101, dtype=np.float64)
    input_image_canny_edge = feature.canny(img)

    input_image_sobel_edge = matlab_sobel_edge(img)
    width = marziliano_method(input_image_sobel_edge, img)
    # print width
    for i in range(1, max_blk_row_idx+1):
        for j in range(1, max_blk_col_idx+1):
            rows = slice(rb*(i-1), rb*i)
            cols = slice(rc*(j-1), rc*j)
            decision = get_edge_blk_decision(input_image_canny_edge[rows, cols], threshold)
            if decision == 1:
                local_width = width[rows, cols]
                local_width = local_width[np.nonzero(local_width)]
                blk_contrast = block_process(img[rows, cols], [rb, rc]) + 1
                blk_jnb = widthjnb[0, int(blk_contrast)-1]
                prob_blur_detection = 1 - math.e ** (-np.power(np.abs(np.true_divide(local_width, blk_jnb)), beta))
                for k in range(1, local_width.size+1):
                    temp_index = int(round(prob_blur_detection[k-1] * 100)) + 1
                    hist_pblur[temp_index-1] = hist_pblur[temp_index-1] + 1
                    total_num_edges = total_num_edges + 1
    if total_num_edges != 0:
        hist_pblur = hist_pblur / total_num_edges
    else:
        hist_pblur = np.zeros(hist_pblur.shape)
    sharpness_metric = np.sum(hist_pblur[0:63])
    return sharpness_metric


def marziliano_method(E, A):
    # print E
    edge_with_map = np.zeros(A.shape)
    gy, gx = np.gradient(A)
    M, N = A.shape
    angle_A = np.zeros(A.shape)
    for m in range(1, M+1):
        for n in range(1, N+1):
            if gx[m-1, n-1] != 0:
                angle_A[m-1, n-1] = math.atan2(gy[m-1,n-1], gx[m-1,n-1]) * (180/np.pi)
            if gx[m-1, n-1] == 0 and gy[m-1, n-1] == 0:
                angle_A[m-1, n-1] = 0
            if gx[m-1, n-1] == 0 and gy[m-1, n-1] == np.pi/2:
                angle_A[m-1, n-1] = 90
    if angle_A.size != 0:
        angle_Arnd = 45 * np.round(angle_A/45.0)
        # print angle_Arnd
        count = 0
        for m in range(2, M):
            for n in range(2, N):
                if E[m-1, n-1] == 1:
                    if angle_Arnd[m-1, n-1] == 180 or angle_Arnd[m-1, n-1] == -180:
                        count += 1
                        for k in range(0, 101):
                            posy1 = n-1-k
                            posy2 = n - 2 - k
                            if posy2 <= 0:
                                break
                            if A[m-1, posy2-1] - A[m-1, posy1-1] <= 0:
                                break
                        width_count_side1 = k + 1
                        for k in range(0, 101):
                            negy1 = n + 1 + k
                            negy2 = n + 2 + k
                            if negy2 > N:
                                break
                            if A[m-1, negy2-1] > A[m-1, negy1-1]:
                                break
                        width_count_side2 = k + 1
                        edge_with_map[m-1, n-1] = width_count_side1 + width_count_side2
                    elif angle_Arnd[m-1, n-1] == 0:
                        count += 1
                        for k in range(0, 101):
                            posy1 = n+1+k
                            posy2 = n + 2 + k
                            if posy2 > N:
                                break
                            # print m, posy2
                            if A[m-1, posy2-1] <= A[m-1, posy1-1]:
                                break
                        width_count_side1 = k + 1
                        for k in range(0, 101):
                            negy1 = n -1-k
                            negy2 = n -2 -k
                            if negy2 <=0:
                                break
                            if A[m-1, negy2-1] >= A[m-1, negy1-1]:
                                break
                        width_count_side2 = k + 1
                        edge_with_map[m-1, n-1] = width_count_side1 + width_count_side2
    return edge_with_map


def get_edge_blk_decision(im_in, T):
    m, n = im_in.shape
    L = m * n
    im_edge_pixels = np.sum(im_in)
    im_out = im_edge_pixels > (L * T)
    return im_out


def matlab_sobel_edge(img):
    mask = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8.0
    bx = correlate(img, mask)
    b = bx*bx
    # print b
    b = b > 4.0
    return np.array(b, dtype=np.int)
################################################################################################



def generating_landmark_lips(test_inf):
    # image = cv2.imread(os.path.join(config.sample_dir,'bg.jpg'))
    # image_real = image.copy()
    # image_fake = image.copy()
    # original = np.array([181,237])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../basics/shape_predictor_68_face_landmarks.dat')

    for inx in range(len(test_inf)):
        real_paths = test_inf[inx]["real_path"]
        fake_paths = test_inf[inx]["fake_path"]
        # print len(real_paths)
        print ("{}/{}".format(inx, len(test_inf)))
        for i in range(len(real_paths)):
            rp = real_paths[i]
            fp = fake_paths[i]
            # print fp
            temp_r = rp.split('/')
            # temp_f = fp.split('/')
            if not os.path.exists( os.path.join(config.sample_dir,'landmark/real64/' + temp_r[-2])):
                os.mkdir( os.path.join(config.sample_dir,'landmark/real64/' + temp_r[-2]))
            if not os.path.exists(os.path.join(config.sample_dir,'landmark/fake64/' + temp_r[-2])):
                os.mkdir(os.path.join(config.sample_dir,'landmark/fake64/' + temp_r[-2]))
            lm_r = os.path.join(config.sample_dir,'landmark/real64/' + temp_r[-2] + '/' + temp_r[-1][:-4] + '.npy' )
            lm_f = os.path.join(config.sample_dir,'landmark/fake64/' + temp_r[-2] + '/' + temp_r[-1][:-4] + '.npy' )
            i_lm_r = os.path.join(config.sample_dir,'landmark/real64/' + temp_r[-2] + '/' + temp_r[-1][:-4] + '.jpg' )
            i_lm_f = os.path.join(config.sample_dir,'landmark/fake64/' + temp_r[-2] + '/' + temp_r[-1][:-4] + '.jpg' )
            image_real = cv2.imread(rp)
            image_fake = cv2.imread(fp)
            # image_real[237:301,181:245,:] = real_mask

            # image_fake[237:301,181:245,:] = fake_mask

            real_gray = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
            real_rects = detector(real_gray, 1)
            fake_gray = cv2.cvtColor(image_fake,cv2.COLOR_BGR2GRAY)
            fake_rects = detector(fake_gray, 1)
            if real_rects is None or fake_rects is None:
                print '--------------------------------'
            for (i,rect) in enumerate(fake_rects):
                shape = predictor(fake_gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    # print name
                    if name != 'mouth':
                        continue
                    clone = image_fake.copy()
                    for (x, y) in shape[i:j]:
                        cv2.circle(clone, (x, y), 1, (0, 255, 0), -1)
                    cv2.imwrite(i_lm_f, clone)

                    mouth_land = shape[i:j].copy()
                    original = np.sum(mouth_land,axis=0) / 20.0
                    # print (mouth_land)
                    mouth_land = mouth_land - original
                    np.save(lm_f,mouth_land)
            for (i,rect) in enumerate(real_rects):
                shape = predictor(real_gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    # print name
                    if name != 'mouth':
                        continue
                    clone = image_real.copy()
                    for (x, y) in shape[i:j]:
                        cv2.circle(clone, (x, y), 1, (0, 255, 0), -1)
                    cv2.imwrite(i_lm_r, clone)
                    mouth_land = shape[i:j].copy()
                    original = np.sum(mouth_land,axis=0) / 20.0

                    # print (mouth_land)
                    mouth_land = mouth_land - original
                    np.save(lm_r,mouth_land) 
def generate_landmarks(pickle_path):
    num_thread = config.num_thread
    test_inf = pickle.load(open(pickle_path, "rb"))
    datas = []
    batch_size = len(test_inf)/num_thread
    temp = []

    if not os.path.exists( os.path.join(config.sample_dir,'landmark')):
        os.mkdir( os.path.join(config.sample_dir,'landmark'))
    if not os.path.exists( os.path.join(config.sample_dir,'landmark/real64')):
        os.mkdir( os.path.join(config.sample_dir,'landmark/real64'))
    if not os.path.exists( os.path.join(config.sample_dir,'landmark/fake64')):
        os.mkdir( os.path.join(config.sample_dir,'landmark/fake64'))
    for i,d in enumerate(test_inf):
        temp.append(d)
        if (i+1) % batch_size ==0:
            datas.append(temp)
            temp = []

    for i in range(num_thread):
        process = multiprocessing.Process(target = generating_landmark_lips,args = (datas[i],))
        process.start()

def compare_landmarks(path, config):
    fake_path = os.path.join(path + 'fake64')

    real_path = os.path.join(path + 'real64')
    # fakes = os.walk(fake_path)
    rps = []
    fps = []
    for video in os.listdir(fake_path):
        for name in os.listdir( os.path.join(fake_path, video) ):
            if name[-3:] == 'npy':
                if os.path.exists(os.path.join(real_path , video, name)):                
                    rps.append(os.path.join(real_path , video, name))
                    fps.append(os.path.join(fake_path , video, name))
    # for root, dirs, files in os.walk(fake_path):
    #     for name in files:
    #         # print name
    #         if name[-3:] == 'npy':
    #             print ('===')
    #             print (os.path.join(root, name))
    #             fps.append(os.path.join(root, name))

                # rps.append(real_path +  '/' + config.model_name + '/' + name)
                # fps.append(fake_path + '/'  +config.model_name  + '/'  + name)
    # for root, dirs, files in os.walk(real_path):
    #     for name in files:
    #         # print name
    #         if name[-3:] == 'npy':
    #             # print ('=++=')
    #             # print (os.path.join(root, name))
    #             rps.append(os.path.join(root, name))

    print (len(rps))
    print (len(fps))
    dis_txt = open(path + 'distance.txt','w')
    distances = []
    # print len(rps)
    for inx in range(len(rps)):
        # try:
            rp = np.load(rps[inx])
            fp = np.load(fps[inx])
            # print rp.shape
            # print fp.shape
            dis = (rp-fp)**2
            dis = np.sum(dis,axis=1)
            dis = np.sqrt(dis)

            dis = np.sum(dis,axis=0)
            distances.append(dis)
            dis_txt.write(rps[inx] + '\t' + str(dis) + '\n')
        # except:

        #     continue 
    average_distance = sum(distances) / len(rps)
    print average_distance

def psnr_f(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def compare_ssim(pickle_path):
    test_inf = pickle.load(open(pickle_path, "rb"))
    dis_txt = open(config.sample_dir + 'ssim.txt','w')
    ssims = []
    # msssims = []
    psnrs =[]
    for i,d in enumerate(test_inf):
        print (i)
        # try:
        fake_paths = d['fake_path']
        real_paths = d['real_path']
        for inx in range(len(fake_paths)):
            f_i = cv2.imread(fake_paths[inx])
            r_i = cv2.imread(real_paths[inx])
            
            r_i = cv2.resize(r_i, (256,256), interpolation = cv2.INTER_AREA)
            f_i = cv2.resize(f_i, (256,256), interpolation = cv2.INTER_AREA)
            ssim = ssim_f(f_i,r_i,multichannel=True)
            f_i = cv2.cvtColor(f_i, cv2.COLOR_BGR2GRAY)
            r_i = cv2.cvtColor(r_i, cv2.COLOR_BGR2GRAY)
            psnr = psnr_f(f_i,r_i)
            # msssim = msssim_f(f_i,r_i)


            psnrs.append(psnr)
            ssims.append(ssim)
            # print "ssim: {:.4f},\t psnr: {:.4f}\t msssim: {:.4f}".format( ssim, psnr,ssim)


            dis_txt.write(fake_paths[inx] + "\t ssim: {:.4f},\t psnr: {:.4f}\t msssim: {:.4f}".format( ssim, psnr,ssim) + '\n') 
        # except:
        #     print ('gggg')
        #     continue
    average_ssim = sum(ssims) / len(ssims)
    average_psnr = sum(psnrs) / len(psnrs)
    # average_msssim = sum(msssims) / len(msssims)
    print "Aeverage: \t ssim: {:.4f},\t psnr: {:.4f}".format( average_ssim, average_psnr,average_ssim)
    return  average_ssim, average_psnr

def compare_cpdb(pickle_path):
    test_inf = pickle.load(open(pickle_path, "rb"))
    dis_txt = open(config.sample_dir + 'cpdb.txt','w')
    r_cpdb = []
    f_cpdb = []
    for i,d in enumerate(test_inf):
        fake_paths = d['fake_path']
        # real_paths = d['real_path']
        for inx in range(len(fake_paths)):
            # real_cpdb = cpbd_compute(real_paths[inx])
            fake_cpdb = cpbd_compute(fake_paths[inx])
            # r_cpdb.append(real_cpdb)
            f_cpdb.append(fake_cpdb)

            dis_txt.write(fake_paths[inx] + '\t fake: {:.4f}'.format(  fake_cpdb) + '\n') 
    # average_r = sum(r_cpdb) / len(r_cpdb)
    average_f = sum(f_cpdb) / len(f_cpdb)
    print "Aeverage: \t fake: {:.4f}".format(  average_f)
    return average_f
def main(config):
    # _sample( config)

    get_pickle(config)
    # get_pickle_xface('/u/lchen63/data/mat/test.csv', config)
    # get_pickle_yousaidthat('/u/lchen63/data/mat/test_yousaidthat.csv', config)
    p = os.path.join( config.pickle_path,'{}.pkl'.format(config.model_name))
    # average_ssim, average_psnr = compare_ssim(p)
    #generate_landmarks(p)
    # average_f = compare_cpdb(p)
    compare_landmarks(os.path.join(config.sample_dir ,'landmark/'), config)
    # print "Aeverage: \t fake: {:.4f}".format(  average_f)
    # print "Aeverage: \t ssim: {:.4f},\t psnr: {:.4f}".format( average_ssim, average_psnr)
if __name__ == "__main__":
    config = parse_args()
    config.sample_dir = os.path.join(config.sample_dir, config.model_name)
    # config.pickle_path = os.path.join(config.pickle_path, config.model_name)
    main(config)
