## VSR PyTorch Implementation
VSR is a collection of recent SR/VSR models, implemented in tensorflow.

This is an **experimental** extension for VSR, porting to [**pytorch**](https://pytorch.org) framework.

## PyTorch v.s. Tensorflow
Why re-invent wheels? This is from my own experience that torch's implementation is always better than tensorflow's (**IMPORTANT: only valid for SR models**).
This extension is used for cross-framework comparison. They share the same data loader, model architecture, benchmark script and even random seeds.

Besides, most works are implemented in pytorch now, it's quite easy to copy their pre-trained models into this framework.

## How to use
Similar to original VSR, there are two main entries "`train.py`" and "`eval.py`"

#### Train models
```bash
python train.py <model-name> [--cuda] [--dataset name] [--epochs num]
```

I.e. `python train.py carn --cuda --dataset=div2k --epochs=1000`.

#### Test models
```bash
python eval.py <model-name> [--cuda] [--test path/or/name] [--pth model.pth/path]
```

I.e. `python eval.py carn --cuda --test=set5 --pth=a/b/c/d.pth`.

For more information about dataset name and more advanced options, please refer to documents of original VSR, [here](../Data/README.md) and [here](../Train/README.md).


## NTIRE 19 Reproduce
Refer to [here](../Docs/README_NTIRE19.md).