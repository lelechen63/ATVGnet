# Hierarchical Cross-modal Talking Face Generation with Dynamic Pixel-wise Loss （ATVGnet）

By [Lele Chen](http://www.cs.rochester.edu/u/lchen63/) ,
[Ross K Maddox](https://www.urmc.rochester.edu/labs/maddox.aspx),
[ Zhiyao Duan](http://www2.ece.rochester.edu/~zduan/),
[Chenliang Xu](https://www.cs.rochester.edu/~cxu22/).

University of Rochester.

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Running](#running)
0. [Model](#model)
0. [Disclaimer and known issues](#disclaimer-and-known-issues)
0. [Results](#results)

### Introduction

This repository contains the original models (AT-net, VG-net) described in the paper "Hierarchical Cross-modal Talking Face Generation with Dynamic Pixel-wise Loss" (https://arxiv.org/abs/1802.02427). This code can be applied directly in [LRW](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) and [GRID](http://spandh.dcs.shef.ac.uk/gridcorpus/). The outputs from the model are visualized here: the first one is the synthesized landmark from ATnet, the rest of them are attention, motion map and final results from VGnet.

![model](https://github.com/lelechen63/ATVGnet/blob/master/img/visualization.gif)


### Citation

If you use these models or the ideas in your research, please cite:
	
	@inproceedings{DBLP:conf/miip/ChenWDAWX18,
	  author    = {Lele Chen and
		       Yue Wu and
		       Adora M. DSouza and
		       Anas Z. Abidin and
		       Axel Wism{\"{u}}ller and
		       Chenliang Xu},
	  title     = {{MRI} tumor segmentation with densely connected 3D {CNN}},
	  booktitle = {Medical Imaging 2018: Image Processing, Houston, Texas, United States,
		       10-15 February 2018},
	  pages     = {105741F},
	  year      = {2018},
	  crossref  = {DBLP:conf/miip/2018},
	  url       = {https://doi.org/10.1117/12.2293394},
	  doi       = {10.1117/12.2293394},
	  timestamp = {Tue, 06 Mar 2018 10:50:01 +0100},
	  biburl    = {https://dblp.org/rec/bib/conf/miip/ChenWDAWX18},
	  bibsource = {dblp computer science bibliography, https://dblp.org}
	}
### Running


0. Pre-installation:[Pytorch 0.4.1](https://pytorch.org/), [dlib](https://pypi.org/project/dlib/), [opencv 2.4.11](https://opencv.org/), [scipy](https://anaconda.org/anaconda/scipy), [librosa](https://librosa.github.io/librosa/), [sk-image](http://scikit-image.org/docs/dev/api/skimage.html), [tqdm](https://github.com/tqdm/tqdm).

0. Download and unzip the training data from [LRW](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

0. Preprocess the data (Extract landmark and crop the image by dlib).
0. Train the ATnet model:  `python aetnet.py`
	- `-device_ids`: gpu id
	- `-batch_size`: batch size 
	- `-model_dir`: folder to save weights
	- `-lstm`:  use lstm or not
	- `-sample_dir`: folder to save visualized images during training
	.


0. Test the model: `python atnet_test.py`
	- `-device_ids`: gpu id
	- `-batch_size`: batch size
	- `-model_name`: pretrained weights
	- `-sample_dir`: folder to save the outputs
	- `-lstm`:  use lstm or not
	- ...

### Model

0. Overall ATVGnet
	![model](https://github.com/lelechen63/ATVGnet/blob/master/img/generator.pdf)

	
0. Regresssion based discriminator network

	![model](https://github.com/lelechen63/ATVGnet/blob/master/img/regress-disc.pdf)

### Disclaimer and known issues

0. These codes are implmented in Tensorflow
0. In this paper, we only use the glioblastoma (HGG) dataset.
0. I didn't config nipype.interfaces.ants.segmentation. So if you need to use `n4correction.py` code, you need to copy it to the bin directory where antsRegistration etc are located. Then run `python n4correction.py`
0. If you want to train these models using this version of tensorflow without modifications, please notice that:
	- You need at lest 12 GB GPU memory.
	- There might be some other untested issues.
	

### Results
0. Result visualization :
	![visualization](https://github.com/lelechen63/MRI-tumor-segmentation-Brats/blob/master/image/h.png)
	![visualization](https://github.com/lelechen63/MRI-tumor-segmentation-Brats/blob/master/image/v.png)

0. Quantitative results:

	model|whole|peritumoral edema (ED)|FGD-enhan. tumor (ET)
	:---:|:---:|:---:|:---:
	Dense24 |0.74| 0.81| 0.80
	Dense48 | 0.61|0.78|0.79
	no-dense|0.61|0.77|0.78
	dense24+n4correction|0.72|0.83|0.81
