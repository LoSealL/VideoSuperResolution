## This is a temporary guide for [ICLR2019 reproducibility challenge](https://reproducibility-challenge.github.io/iclr_2019/)

Review target: [The relativistic discriminator: a key element missing from standard GAN](https://openreview.net/forum?id=S1erHoR5t7) (*Accept Poster*)

Ticket: [S1erHoR5t7](https://github.com/reproducibility-challenge/iclr_2019/issues/10)

### How to reproduce benchmark

1. Download dataset and weights

   1. CIFAR10 downloads automatically
   2. CelebA is downloaded [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and is processed:
      ```bash
      python VSR/Tools/DataProcessing/CelebA.py /mnt/data/dataset/celeba/ --n_test=10000
      ```

   3. weights are downloaded by `python prepare_data.py --filter="\w*gan"`, which will extract weights into `./Results/`

2. Evaluate GAN models

   One sample:
   ```bash
   cd Train
   python run.py --mode=eval --model=rgan --checkpoint_dir=../Results/rgan --epochs=500 --test=cifar10 --enable_inception_score --enable_fid
   ```
   
3. Generate samples

   ```bash
   cd Train
   python run.py --model=rgan --test=cifar10
   ```

4. Train models from scratch

    1. Refer to general guide [here](./README.md)
    2. (Optional) Prepare your own dataset (if needed, refer DDF [here](./Data/README.md))
    3. (Optional) Modify [model config file](./Train/parameters/rgan.yaml), all models and information are defined [here](./VSR/Models/Gan.py)
    4. Run script: (i.e. RGAN)
    ```bash
    cd Train
    python run.py --model=rgan --epochs=500 --dataset=cifar10
    ```