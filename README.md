# Video Super Resolution
A collection of state-of-the-art video or single-image super-resolution architectures, reimplemented in tensorflow.

**Pretrained weights is uploading now.**

**Several referenced PyTorch implementations are also included now.**

**Quick Link:** [Installation](#install) [Getting Started](#Getting-Started) 

## Network list and reference (Updating)
The hyperlink directs to paper site, follows the official codes if the authors open sources.

All these models are implemented in **ONE** framework.

|Model |Published |Code* |VSR (TF)**|VSR (Torch)|Keywords|Pretrained|
|:-----|:---------|:-----|:---------|:----------|:-------|:---------|
|SRCNN|[ECCV14](https://arxiv.org/abs/1501.00092)|-, [Keras](https://github.com/qobilidop/srcnn)|Y|Y| Kaiming |[√](https://github.com/LoSealL/Model/releases)|
|RAISR|[arXiv](https://arxiv.org/abs/1606.01299)|-|-|-| Google, Pixel 3 ||
|ESPCN|[CVPR16](https://arxiv.org/abs/1609.05158)|-, [Keras](https://github.com/qobilidop/srcnn)|Y|Y| Real time |[√](https://github.com/LoSealL/Model/releases)|
|VDSR|[CVPR16](https://arxiv.org/abs/1511.04587)|-|Y|Y| Deep, Residual |[√](https://drive.google.com/open?id=1hW5YDxXpmjO2IfAy8f29O7yf1M3fPIg1)|
|DRCN|[CVPR16](https://arxiv.org/abs/1511.04491)|-|Y|Y| Recurrent ||
|DRRN|[CVPR17](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf)|[Caffe](https://github.com/tyshiwo/DRRN_CVPR17), [PyTorch](https://github.com/jt827859032/DRRN-pytorch)|Y|Y| Recurrent ||
|LapSRN|[CVPR17](http://vllab.ucmerced.edu/wlai24/LapSRN/)|[Matlab](https://github.com/phoenix104104/LapSRN)|Y|-| Huber loss ||
|EDSR|[CVPR17](https://arxiv.org/abs/1707.02921)|-|Y|Y| NTIRE17 Champion |[√](https://github.com/LoSealL/Model/releases)|
|SRGAN|[CVPR17](https://arxiv.org/abs/1609.04802)|-|Y|-| 1st proposed GAN ||
|VESPCN|[CVPR17](https://arxiv.org/abs/1611.05250)|-|Y|Y| VideoSR |[√](https://drive.google.com/open?id=19u4YpsyThxW5dv4fhpMj7c5gZeEDKthm)|
|MemNet|[ICCV17](https://arxiv.org/abs/1708.02209)|[Caffe](https://github.com/tyshiwo/MemNet)|Y|-|||
|SRDenseNet|[ICCV17](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)|-, [PyTorch](https://github.com/wxywhu/SRDenseNet-pytorch)|Y|-| Dense |[√](https://drive.google.com/open?id=1aXAfRqZieY6mTfZUnErG84-9NfkQSeDw)|
|SPMC|[ICCV17](https://arxiv.org/abs/1704.02738)|[Tensorflow](https://github.com/jiangsutx/SPMC_VideoSR)|T|Y| VideoSR ||
|DnCNN|[TIP17](http://ieeexplore.ieee.org/document/7839189/)|[Matlab](https://github.com/cszn/DnCNN)|Y|Y| Denoise |[√](https://github.com/LoSealL/Model/releases)|
|DCSCN|[arXiv](https://arxiv.org/abs/1707.05425)|[Tensorflow](https://github.com/jiny2001/dcscn-super-resolution)|Y|-|||
|IDN|[CVPR18](https://arxiv.org/abs/1803.09454)|[Caffe](https://github.com/Zheng222/IDN-Caffe)|Y|-| Fast |[√](https://drive.google.com/open?id=1Fh3rtvrKKLAK27r518T1M_JET_LWZAFQ)|
|RDN|[CVPR18](https://arxiv.org/abs/1802.08797)|[Torch](https://github.com/yulunzhang/RDN)|Y|-| Deep, BI-BD-DN ||
|SRMD|[CVPR18](https://arxiv.org/abs/1712.06116)|[Matlab](https://github.com/cszn/SRMD)|T|-| Denoise/Deblur/SR ||
|DBPN|[CVPR18](https://arxiv.org/abs/1803.02735)|[PyTorch](https://github.com/alterzero/DBPN-Pytorch)|Y|Y| NTIRE18 Champion |[√](https://drive.google.com/open?id=1ymtlOjhkGmad-od0zw7yTf17nWD4KMVi)|
|ZSSR|[CVPR18](http://www.wisdom.weizmann.ac.il/~vision/zssr/)|[Tensorflow](https://github.com/assafshocher/ZSSR)|-|-| Zero-shot ||
|FRVSR|[CVPR18](https://arxiv.org/abs/1801.04590)|[PDF](https://github.com/msmsajjadi/FRVSR)|T|Y| VideoSR |[√](https://github.com/LoSealL/Model/releases)|
|DUF|[CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf)|[Tensorflow](https://github.com/yhjo09/VSR-DUF)|T|-| VideoSR ||
|CARN|[ECCV18](https://arxiv.org/abs/1803.08664)|[PyTorch](https://github.com/nmhkahn/CARN-pytorch)|Y|Y| Fast |[√](https://github.com/LoSealL/Model/releases/carn)|
|RCAN|[ECCV18](https://arxiv.org/abs/1807.02758)|[PyTorch](https://github.com/yulunzhang/RCAN)|Y|Y| Deep, BI-BD-DN ||
|MSRN|[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf)|[PyTorch](https://github.com/MIVRC/MSRN-PyTorch)|Y|Y| |[√](https://drive.google.com/open?id=1A0LoY3oB_VnArP3GzI1ILUNJbLAEjdtJ)|
|SRFeat|[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf)|[Tensorflow](https://github.com/HyeongseokSon1/SRFeat)|Y|Y| GAN ||
|NLRN|[NIPS18](https://papers.nips.cc/paper/7439-non-local-recurrent-network-for-image-restoration.pdf)|[Tensorflow](https://github.com/Ding-Liu/NLRN)|T|-| Non-local, Recurrent ||
|SRCliqueNet|[NIPS18](https://arxiv.org/abs/1809.04508)|-|-|-| Wavelet ||
|FFDNet|[TIP18](https://ieeexplore.ieee.org/document/8365806/)|[Matlab](https://github.com/cszn/FFDNet)|Y|Y| Conditional denoise||
|CBDNet|[arXiv](https://arxiv.org/abs/1807.04686)|[Matlab](https://github.com/GuoShi28/CBDNet)|T|-| Blind-denoise ||
|SOFVSR|[ACCV18](http://arxiv.org/abs/1809.08573)|[PyTorch](https://github.com/LongguangWang/SOF-VSR)|-|Y| VideoSR |[√](https://github.com/LoSealL/Model/releases/download/sofvsr/SOFVSR_x4.zip)|
|ESRGAN|[ECCVW18](http://arxiv.org/abs/1809.00219)|[PyTorch](https://github.com/xinntao/ESRGAN)|-|Y|1st place PIRM 2018|[√](https://github.com/LoSealL/Model/releases/download/esrgan/esrgan.zip)|
|TecoGAN|[arXiv](http://arxiv.org/abs/1811.09393)|[Tensorflow](https://github.com/thunil/TecoGAN)|-|T| VideoSR GAN|[√](https://github.com/LoSealL/Model/releases/download/tecogan/tecogan.zip)|
|RBPN|[CVPR19](https://arxiv.org/abs/1903.10128)|[PyTorch](https://github.com/alterzero/RBPN-PyTorch)|-|Y| VideoSR ||

\*The 1st repo is by paper author.

\**__Y__: included; __-__: not included; __T__: under-testing. 

You can download pre-trained weights through [`prepare_data`](./prepare_data.py), or visit the hyperlink at **√**.

## Link of datasets
*(please contact me if any of links offend you or any one disabled)*

|Name|Usage|#|Site|Comments|
|:---|:----|:----|:---|:-----|
|SET5|Test|5|[download](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|SET14|Test|14|[download](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|SunHay80|Test|80|[download](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|Urban100|Test|100|[download](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|VID4|Test|4|[download](https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip)|4 videos|
|BSD100|Train|300|[download](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|BSD300|Train/Val|300|[download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz)|-|
|BSD500|Train/Val|500|[download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)|-|
|91-Image|Train|91|[download](http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar)|Yang|
|DIV2K|Train/Val|900|[website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)|NTIRE17|
|Waterloo|Train|4741|[website](https://ece.uwaterloo.ca/~k29ma/exploration/)|-|
|MCL-V|Train|12|[website](http://mcl.usc.edu/mcl-v-database/)|12 videos|
|GOPRO|Train/Val|33|[website](https://github.com/SeungjunNah/DeepDeblur_release)|33 videos, deblur|
|CelebA|Train|202599|[website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)|Human faces|
|Sintel|Train/Val|35|[website](http://sintel.is.tue.mpg.de/downloads)|Optical flow|
|FlyingChairs|Train|22872|[website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)|Optical flow|
|DND|Test|50|[website](https://noise.visinf.tu-darmstadt.de/)|Real noisy photos|
|RENOIR|Train|120|[website](http://ani.stat.fsu.edu/~abarbu/Renoir.html)|Real noisy photos|
|NC|Test|60|[website](http://demo.ipol.im/demo/125/)|Noisy photos|
|SIDD(M)|Train/Val|200|[website](https://www.eecs.yorku.ca/~kamel/sidd/)|NTIRE 2019 Real Denoise|
|RSR|Train/Val|80|[download]()|NTIRE 2019 Real SR|
|Vimeo-90k|Train/Test|89800|[website](http://toflow.csail.mit.edu/)|90k HQ videos|

Other open datasets:
[Kaggle](https://www.kaggle.com/datasets)
[ImageNet](http://www.image-net.org/)
[COCO](http://cocodataset.org/)

## VSR package
This package offers a training and data processing framework based on [TF](https://www.tensorflow.org).
What I made is a simple, easy-to-use framework without lots of encapulations and abstractions.
Moreover, VSR can handle raw NV12/YUV as well as a sequence of images as inputs.

### Install

1. Prepare proper tensorflow and pytorch(optional). For example, GPU and CUDA10.0 (recommend to use `conda`):

   ```shell
   conda install tensorflow-gpu==1.12.0
   # optional
   # conda install pytorch==1.0.1
   ```

2. Install VSR package

   ```bash
   # For someone see this doc online
   # git clone https://github.com/loseall/VideoSuperResolution && cd VideoSuperResolution
   pip install -e .
   ```
### Getting Started

1. Download pre-trained weights and (optinal) training datasets. For instance, let\'s begin with VESPCN and vid4 test data:
   ```shell
   python prepare_data.py --filter vespcn vid4
   ```

2. Evaluate
   ```shell
   cd Train
   python run.py --model vespcn --test vid4
   ```

3. Train
   ```shell
   python prepare_data.py --filter mcl-v
   cd Train
   python run.py --model vespcn --dataset mcl-v --memory_limit 1GB --epochs 100
   ```

OK, that's all you need. For more details, use `--help` to get more information.

To train/test/infer any other model in [VSR.Models](./VSR/Models/__init__.py), please see [README](./Train/README.md).
To write and train your own model via VSR, please see [Docs](./Docs).
