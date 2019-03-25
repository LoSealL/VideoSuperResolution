## VSR PyTorch Implementation
VSR is a collection of recent SR/VSR models, implemented in tensorflow.

This is an **experimental** extension for VSR, porting to [**pytorch**](https://pytorch.org) framework.

## PyTorch v.s. Tensorflow
Why re-invent wheels? This is from my own experience that torch's implementation is always better than tensorflow's (**IMPORTANT: only confirmed on SISR models**).
This extension is used for cross-framework comparison. They share the same data loader, model architecture, benchmark script and even random seeds.

Besides, most works are implemented in pytorch now, it's quite easy to integrate their pre-trained models into this framework.

## Getting Started
Similar to original VSR, there are two main entries "`train.py`" and "`eval.py`"

#### Train models
```bash
python train.py <model-name> [--cuda] [--dataset name] [--epochs num]
```

#### Test models
```bash
python eval.py <model-name> [--cuda] [--test path/or/name] [--pth model.pth/path]
```

For more information about dataset name and more advanced options, please refer to documents of original VSR, [here](../Data/README.md) and [here](../Train/README.md).

#### Sample Code: CARN
```bash
# training
python train.py carn --cuda --dataset div2k --epochs 1000
# testing
python eval.py carn --cuda --test set5 set14 bsd100 urban100
# predicting own data
python eval.py carn --cuda --test /tmp/myphotos/*.png
```


## NTIRE 19 Reproduce
Refer to [here](../Docs/README_NTIRE19.md).