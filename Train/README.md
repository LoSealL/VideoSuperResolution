# Model Tools
VSR provides **2** tools along with code repo, that help users train and test models.

To select a proper backend as you wish, please read [Change-backend](../Docs/HowTo/Change-backend.md).

## 1. Train
### Usage: `py train.py -h`
```bash
usage: train.py [-h] [-p PARAMETER] [--save_dir SAVE_DIR]
                [--data_config DATA_CONFIG] [--dataset DATASET]
                [--epochs EPOCHS] [--steps STEPS] [--val_steps VAL_STEPS]
                [--cuda] [--threads THREADS] [--memory_limit MEMORY_LIMIT]
                [--traced_val] [--pretrain PRETRAIN] [--export EXPORT]
                [-c COMMENT]
                {espcn,srcnn,vdsr,dncnn,drcn,drrn,ffdnet,edsr,carn,dbpn,rcan,srfeat,esrgan,msrn,crdn,mldn,drn,sofvsr,vespcn,frvsr,qprn,ufvsr,yovsr,tecogan,spmc,rbpn,didn,dhdn,grdn,resunet,edrn,frn,ran}
```
The bracket {} lists all supported models in the current backend.

### Dataset
The dataset used in this tool is described in [datasets.yaml](../Data/datasets.yaml).
You can read [README.md](../Data/README.md) in that folder for details.

### Samples
```bash
python train.py srcnn --dataset 91image --epochs 100 --cuda
```

If your system memory is not enough to hold all training data, and you want to running other process, add `--memory_limit=???` to constrain the memory usage (roughly).
```bash
python train.py vdsr --dataset div2k --epochs 100 --cuda --memory_limit=4GB
```

If you want to continue to train from an external checkpoint, you can explicitly specify the checkpoint by adding `--pretrain=<path>`.
```bash
python train.py carn --dataset div2k --epochs 100 --cuda --pretrain=/model/carn/carn.pth
```


## 2. Evaluate
Evaluation is almost the same as training.
### Usage: `py eval.py -h`
```bash
usage: eval.py [-h] [-p PARAMETER] [-t [TEST [TEST ...]]]
               [--save_dir SAVE_DIR] [--data_config DATA_CONFIG]
               [--pretrain PRETRAIN] [--ensemble] [--video] [--cuda]
               [--threads THREADS] [--output_index OUTPUT_INDEX]
               [--auto_rename] [-c COMMENT]
               {espcn,srcnn,vdsr,dncnn,drcn,drrn,ffdnet,edsr,carn,dbpn,rcan,srfeat,esrgan,msrn,crdn,mldn,drn,sofvsr,vespcn,frvsr,qprn,ufvsr,yovsr,tecogan,spmc,rbpn,didn,dhdn,grdn,resunet,edrn,frn,ran}
```
The bracket {} lists all supported models in the current backend.

### Flag `-t`
You can specify a named dataset through `-t`, or an existing path to `-t`.
`-t` also supports a list of arguments.
```bash
python eval.py vdsr --cuda -t set5 set14
```
```bash
python eval.py vdsr --cuda -t /data/datasets/test/my-photos/*01.png
python eval.py vdsr --cuda -t /data/datasets/test/bsd100 --pretrain=/model/vdsr/vdsr.pth
```

### Flag `--video`
If you are evaluating a video SR model, to read external data as video stream, parse `--video`:
```bash
python eval.py vespcn --cuda -t /data/video/my-videos --video
```
However, if evaluate with a named dataset, `--video` is not needed:
```bash
# vid4 in datasets.yaml is tagged with [video]
python eval.py vespcn --cuda -t vid4
```

## 3. Changing model parameters
All model parameters are in [par](./par/).
You can overwrite model parameters by adding arguments accordingly.

For example, `srcnn` has parameters under [srcnn.yml](./par/pytorch/srcnn.yml).
```yaml
srcnn:
  scale: 4
  channel: 3
  filters: [9, 5, 5]
  upsample: true
  image_weight: 1
  feature_weight: 0  # perceptual loss
```
Then you can overwrite `channel` by:
```bash
python train.py srcnn --cuda --dataset 91image --epochs 200 --channel 1
```
