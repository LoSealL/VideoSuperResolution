# Video Super Resolution
A collection of state-of-the-art video or single-image super-resolution architectures, reimplemented in tensorflow.
(Now VEPSCN model has finally been added to Models, pretrained weights will be uploaded soon...)

## Network list and reference (Updating)
The hyperlink directs to paper site, follows the official codes if the authors open sources.

All these models are implemented in **ONE** framework.

|Model |Published |Code* |VSR Included**|
|:-----|:---------|:----|:-----------|
|SRCNN|[ECCV14](https://arxiv.org/abs/1501.00092)|-, [Keras](https://github.com/qobilidop/srcnn)|Y|
|RAISR|[arXiv](https://arxiv.org/abs/1606.01299)|-|N|
|ESPCN|[CVPR16](https://arxiv.org/abs/1609.05158)|-, [Keras](https://github.com/qobilidop/srcnn)|Y|
|VDSR|[CVPR16](https://arxiv.org/abs/1511.04587)|-|Y|
|DRCN|[CVPR16](https://arxiv.org/abs/1511.04491)|-|Y|
|DRRN|[CVPR17](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf)|[Caffe](https://github.com/tyshiwo/DRRN_CVPR17), [PyTorch](https://github.com/jt827859032/DRRN-pytorch)|Y|
|LapSRN|[CVPR17](http://vllab.ucmerced.edu/wlai24/LapSRN/)|[Matlab](https://github.com/phoenix104104/LapSRN)|Y|
|EDSR|[CVPR17](https://arxiv.org/abs/1707.02921)|-|Y|
|MemNet|[ICCV17](https://arxiv.org/abs/1708.02209)|[Caffe](https://github.com/tyshiwo/MemNet)|Y|
|SRDenseNet|[ICCV17](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)|-, [PyTorch](https://github.com/wxywhu/SRDenseNet-pytorch)|T|
|SPMC|[ICCV17](https://arxiv.org/abs/1704.02738)|[Tensorflow](https://github.com/jiangsutx/SPMC_VideoSR)|N|
|DnCNN|[TIP17](http://ieeexplore.ieee.org/document/7839189/)|[Matlab](https://github.com/cszn/DnCNN)|Y|
|DCSCN|[arXiv](https://arxiv.org/abs/1707.05425)|[Tensorflow](https://github.com/jiny2001/dcscn-super-resolution)|Y|
|IDN|[CVPR18](https://arxiv.org/abs/1803.09454)|[Caffe](https://github.com/Zheng222/IDN-Caffe)|Y|
|RDN|[CVPR18](https://arxiv.org/abs/1802.08797)|[Torch](https://github.com/yulunzhang/RDN)|Y|
|SRMD|[CVPR18](https://arxiv.org/abs/1712.06116)|[Matlab](https://github.com/cszn/SRMD)|T|
|DBPN|[CVPR18](https://arxiv.org/abs/1803.02735)|[PyTorch](https://github.com/alterzero/DBPN-Pytorch)|Y|
|ZSSR|[CVPR18](http://www.wisdom.weizmann.ac.il/~vision/zssr/)|[Tensorflow](https://github.com/assafshocher/ZSSR)|N|
|FRVSR|[CVPR18](https://arxiv.org/abs/1801.04590)|[PDF](https://github.com/msmsajjadi/FRVSR)|T|
|DUF|[CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf)|[Tensorflow](https://github.com/yhjo09/VSR-DUF)|T|
|CARN|[ECCV18](https://arxiv.org/abs/1803.08664)|[PyTorch](https://github.com/nmhkahn/CARN-pytorch)|Y|
|RCAN|[ECCV18](https://arxiv.org/abs/1807.02758)|[PyTorch](https://github.com/yulunzhang/RCAN)|Y|
|MSRN|[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf)|[PyTorch](https://github.com/MIVRC/MSRN-PyTorch)|T|
|SRFeat|[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf)|[Tensorflow](https://github.com/HyeongseokSon1/SRFeat)|T|
|NLRN|[NIPS18](https://papers.nips.cc/paper/7439-non-local-recurrent-network-for-image-restoration.pdf)|[Tensorflow](https://github.com/Ding-Liu/NLRN)|T|
|SRCliqueNet|[NIPS18](https://arxiv.org/abs/1809.04508)|-|N|

\*The 1st repo is by paper author.

\**__Y__: included; __N__: non-included; __T__: under-testing. 


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
|BSD300|Train|300|[download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz)|-|
|BSD500|Train|500|[download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)|-|
|91-Image|Train|91|[download](http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar)|Yang|
|DIV2K|Train|900|[website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)|NTIRE17|
|Waterloo|Train|4741|[website](https://ece.uwaterloo.ca/~k29ma/exploration/)|-|
|MCL-V|Train|12|[website](http://mcl.usc.edu/mcl-v-database/)|12 videos|
|GOPRO|Train|33|[website](https://github.com/SeungjunNah/DeepDeblur_release)|33 videos|


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
