A collection of state-of-the-art video or single-image super-resolution architectures, reimplemented in tensorflow.
(currently no video models...)

## Network list and reference
The hyperlink directs to paper site, follows the official codes if the authors open sources.
- Classic

  1. [**SRCNN**](https://arxiv.org/abs/1501.00092)
- CVPR 2016

  2. Efficient Sub-Pixel Convolutional Network: [**ESPCN**](https://arxiv.org/abs/1609.05158)
  3. Very Deep Convolutional Networks: [**VDSR**](https://arxiv.org/abs/1511.04587)
  4. Deeply-Recursive Convolutional Network: [**DRCN**](https://arxiv.org/abs/1511.04491)
- CVPR 2017

  5. Deep Recursive Residual Network: [**DRRN**](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) *https://github.com/tyshiwo/DRRN_CVPR17*
  6. Deep Laplacian Pyramid Networks: [**LapSRN**](http://vllab.ucmerced.edu/wlai24/LapSRN/) *https://github.com/phoenix104104/LapSRN*
  7. Enhanced Deep Residual Networks: [**EDSR**](https://arxiv.org/abs/1707.02921)
- ICCV 2017

  8. Memory Network: [**MemNet**](https://arxiv.org/abs/1708.02209) *https://github.com/tyshiwo/MemNet*
- CVPR 2018

  9. Information Distillation Network: [**IDN**](https://arxiv.org/abs/1803.09454) *https://github.com/Zheng222/IDN-Caffe*
  10. Residual Dense Network: [**RDN**](https://arxiv.org/abs/1802.08797) *https://github.com/yulunzhang/RDN*
  11. Super-Resolution Network for Multiple Degradations: [**SRMD**](https://arxiv.org/abs/1712.06116) *https://github.com/cszn/SRMD*
  12. Deep Back-Projection Networks: [**DBPN**](https://arxiv.org/abs/1803.02735) https://github.com/alterzero/DBPN-Pytorch
  13. Zero-Shot Super-Resolution: [**ZSSR**](http://www.wisdom.weizmann.ac.il/~vision/zssr/) https://github.com/assafshocher/ZSSR
- ECCV 2018

  14. Cascading Residual Network: [**CARN**](https://arxiv.org/abs/1803.08664) https://github.com/nmhkahn/CARN-pytorch
  15. Residual Channel Attention Networks: [**RCAN**](https://arxiv.org/abs/1807.02758) https://github.com/yulunzhang/RCAN
- Others

  - [**DNCNN**](http://ieeexplore.ieee.org/document/7839189/) (*This is for denoise*) *https://github.com/cszn/DnCNN*
  - Deep CNN
with Skip Connection: [**DCSCN**](https://arxiv.org/abs/1707.05425) *https://github.com/jiny2001/dcscn-super-resolution*

- **Videos**

  - CVPR2017 [**VESPCN**](https://arxiv.org/abs/1611.05250)
  - ICCV2017 [**SPMC**](https://arxiv.org/abs/1704.02738)  https://github.com/jiangsutx/SPMC_VideoSR
  - CVPR2018 [**FRVSR**](https://arxiv.org/abs/1801.04590) https://github.com/msmsajjadi/FRVSR
  - CVPR2018 [**DUF**](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf) https://github.com/yhjo09/VSR-DUF
  
All these models are implemented in **ONE** framework.

## VSR package
This package offers a training and data processing framework based on [TF](https://www.tensorflow.org).
What I made is a simple, easy-to-use framework without lots of encapulations and abstractions.
Moreover, VSR can handle raw NV12/YUV as well as a sequence of images as inputs.

### Install
```bash
git clone https://github.com/loseall/VideoSuperResolution && cd VideoSuperResolution
pip install -e .
```
**Require**: tensorflow, numpy, PIL, pypng, psutil, pytest, tqdm

### DataLoader
- `Dataset` offers manipulation of **virtual** images.
**Virtual** means the images can be either a single png file or a list of jpeg files as a sequence of images.
One need to provide a JSON config file, see [here](./Data/datasets.json).
- `Loader` offers `BatchLoader` object to generate image patches with HR and LR pairs
- `VirtualFile` is internally used object to depict the **file** object

### Framework
- `Environment` offers a simple framework specific to super resolution. See [examples](./UTest/train_srcnn.py) for instance.
And see [Environment.py](./VSR/Framework/Environment.py) for details.
- `Callbacks` offers a collection of callback functions used in `Environment.fit`, `Environment.test` and `Environment.predict`
- `SuperResolution` is the parent object to all models

### Models
Offers a collection of implementations for recent papers and research works.

## Data
To avoid storing a mess of images in codebase, I offer you links to widely used database and a configuration file to
describe your own datasets.
For config file, see [here](./Data/datasets.json) as a sample, and [here](./Data/README.md) for details.

### list of datasets
- [SET 5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
- [SET 14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)
- [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)
- [SunHays80](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip)
- [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip)

*Above links are from [jbhuang0604](https://github.com/jbhuang0604/SelfExSR)*

- [BSD300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz)
- [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)
- [91-Image](http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar)
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [Waterloo](https://ece.uwaterloo.ca/~k29ma/dataset/exploration_database_and_code.rar)
- [MCL-V](http://mcl.usc.edu/mcl-v-database/) *This is video datasets*
- [VID4](https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip) *Video test set* 

## Training
You can either train via `Environment` object or via your own script.
You can also use [pre-made](./Train/train.py) script to train the models in VSR package.
See [readme](./Train/README.md) for details.

## Todo
- [ ] MemNet
- [ ] RCAN
- [ ] CARN
- [ ] FRVSR