# Video Super Resolution
A collection of state-of-the-art video or single-image super-resolution architectures, reimplemented in tensorflow.
(Now VEPSCN model has finally been added to Models, pretrained weights will be uploaded soon...)

## Network list and reference
The hyperlink directs to paper site, follows the official codes if the authors open sources.
- Classic

  1. [**SRCNN**](https://arxiv.org/abs/1501.00092)
  1. [**RAISR**](https://arxiv.org/abs/1606.01299)
  
- CVPR 2016

  1. Efficient Sub-Pixel Convolutional Network: [**ESPCN**](https://arxiv.org/abs/1609.05158)
  1. Very Deep Convolutional Networks: [**VDSR**](https://arxiv.org/abs/1511.04587)
  1. Deeply-Recursive Convolutional Network: [**DRCN**](https://arxiv.org/abs/1511.04491)
  
- CVPR 2017

  1. Deep Recursive Residual Network: [**DRRN**](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf)
  [code](https://github.com/tyshiwo/DRRN_CVPR17)
  1. Deep Laplacian Pyramid Networks: [**LapSRN**](http://vllab.ucmerced.edu/wlai24/LapSRN/)
  [code](https://github.com/phoenix104104/LapSRN)
  1. Enhanced Deep Residual Networks: [**EDSR**](https://arxiv.org/abs/1707.02921)
  
- ICCV 2017

  1. Memory Network: [**MemNet**](https://arxiv.org/abs/1708.02209)
   [code](https://github.com/tyshiwo/MemNet)
   
- CVPR 2018

  1. Information Distillation Network: [**IDN**](https://arxiv.org/abs/1803.09454)
   [code](https://github.com/Zheng222/IDN-Caffe)
  1. Residual Dense Network: [**RDN**](https://arxiv.org/abs/1802.08797)
   [code](https://github.com/yulunzhang/RDN)
  1. Super-Resolution Network for Multiple Degradations: [**SRMD**](https://arxiv.org/abs/1712.06116)
   [code](https://github.com/cszn/SRMD)
  1. Deep Back-Projection Networks: [**DBPN**](https://arxiv.org/abs/1803.02735)
   [code](https://github.com/alterzero/DBPN-Pytorch)
  1. Zero-Shot Super-Resolution: [**ZSSR**](http://www.wisdom.weizmann.ac.il/~vision/zssr/)
   [code](https://github.com/assafshocher/ZSSR)
   
- ECCV 2018

  1. Cascading Residual Network: [**CARN**](https://arxiv.org/abs/1803.08664)
   [code](https://github.com/nmhkahn/CARN-pytorch)
  1. Residual Channel Attention Networks: [**RCAN**](https://arxiv.org/abs/1807.02758)
   [code](https://github.com/yulunzhang/RCAN)
   
- Others

  1. [**DNCNN**](http://ieeexplore.ieee.org/document/7839189/) (*This is for denoise*)
   [code](https://github.com/cszn/DnCNN)
  1. Deep CNN
with Skip Connection: [**DCSCN**](https://arxiv.org/abs/1707.05425)
   [code](https://github.com/jiny2001/dcscn-super-resolution)

- **Videos**

  1. [**VESPCN**](https://arxiv.org/abs/1611.05250)
  1. [**SPMC**](https://arxiv.org/abs/1704.02738)
  [code](https://github.com/jiangsutx/SPMC_VideoSR)
  1. [**FRVSR**](https://arxiv.org/abs/1801.04590)
  [code](https://github.com/msmsajjadi/FRVSR)
  1. [**DUF**](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf)
  [code](https://github.com/yhjo09/VSR-DUF)
  
All these models are implemented in **ONE** framework.

## Link of datasets
*(please contact me if any of links offend you or any one disabled)*
- [SET 5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
- [SET 14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)
- [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)
- [SunHays80](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip)
- [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip)

[THANKS] *Above links are from [jbhuang0604](https://github.com/jbhuang0604/SelfExSR)*

- [BSD300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz)
- [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)
- [91-Image](http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar)
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [Waterloo](https://ece.uwaterloo.ca/~k29ma/exploration/)
- [MCL-V](http://mcl.usc.edu/mcl-v-database/)
- [GOPRO](https://github.com/SeungjunNah/DeepDeblur_release)
- [VID4](https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip)

## VSR package
This package offers a training and data processing framework based on [TF](https://www.tensorflow.org).
What I made is a simple, easy-to-use framework without lots of encapulations and abstractions.
Moreover, VSR can handle raw NV12/YUV as well as a sequence of images as inputs.

### Install
```bash
git clone https://github.com/loseall/VideoSuperResolution && cd VideoSuperResolution
pip install -e .
```

### How to use
To train/test/infer any model in [VSR.Models](./VSR/Models/__init__.py), please see [README](./Train/README.md).
To write and train your own model via VSR, please see [Docs](./Docs).

### Todo
- [ ] FRVSR
- [ ] ZSSR