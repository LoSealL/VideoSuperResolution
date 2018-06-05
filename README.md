A collection of state-of-the-art video or single-image super-resolution architectures, reimplemented in tensorflow.
(currently no video models...)

## Network list and reference
The hyperlink directs to paper site, follows the official codes if the authors open sources.
- Classic
	1. [**SRCNN**](https://arxiv.org/abs/1501.00092)
- CVPR 2016
	2. [**ESPCN**](https://arxiv.org/abs/1609.05158)
	3. [**VDSR**](https://arxiv.org/abs/1511.04587)
	4. [**DRCN**](https://arxiv.org/abs/1511.04491)
- CVPR 2017
	5. [**DRRN**](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) *https://github.com/tyshiwo/DRRN_CVPR17*
	6. [**LapSRN**](http://vllab.ucmerced.edu/wlai24/LapSRN/) *https://github.com/phoenix104104/LapSRN*
- ICCV 2017
	7. [**MemNet**](https://arxiv.org/abs/1708.02209) *https://github.com/tyshiwo/MemNet*
- CVPR 2018
	8. [**IDN**](https://arxiv.org/abs/1803.09454) *https://github.com/Zheng222/IDN-Caffe*
	9. [**RDN**](https://arxiv.org/abs/1802.08797) *https://github.com/yulunzhang/RDN*
	10. [**SRMD**](https://arxiv.org/abs/1712.06116) *https://github.com/cszn/SRMD*
- Others
	11. [**DNCNN**](http://ieeexplore.ieee.org/document/7839189/) (*This is for denoise*) *https://github.com/cszn/DnCNN*
	12. [**DCSCN**](https://arxiv.org/abs/1707.05425) *https://github.com/jiny2001/dcscn-super-resolution*

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
**Require**: tensorflow-gpu, numpy, PIL

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
- [SET 5]()
- [SET 14]()
- [BSD100]()
- [BSD200]()
- [BSD500]()
- [91-Image]()
- [SunHays80]()
- [Urban100]()
- [MCL-V]() *This is video datasets*


## Training
You can either train via `Environment` object or via your own script.
You can also use [pre-made](./Train/train.py) script to train the models in VSR package.
See [readme](./Train/README.md) for details.


## Todo
- [x] SRCNN, ESPCN, VDSR, DRCN, IDN, RDN, DNCNN, DCSCN
- [ ] Add links to dataset
- [ ] DRRN
- [ ] LapSRN
- [ ] MemNet