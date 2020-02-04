# Manual **run.py**:
A one-line command to model executor

## 1. Train your 1st model:

```bash
python run.py --model=srcnn --dataset=91-image --test=set14
```
That's all! It will read training data from `91-image` and train `srcnn` network 50 epochs with 50*200 steps.
And test it on `Set14` dataset afterward.

## 2. Train model more customized:
`run.py` can be used to train/benchmark/infer models collected in [VSR.Models](../VSR/Models)

The model parameters can be configured through [parameters/root.yaml](tf/root.yaml) and
*<parameters/model-name.yaml>* (i.e. [srcnn](tf/srcnn.yaml))

```Bash
python run.py --model <model-name> --epochs <num> --steps_per_epoch <num> --dataset <train-dataset-name> \
  --test <test-dataset-name> --infer <infer-dir | single-file | infer-dataset-name> --threads <loading-thread-num>\
  --save_dir <save>
```
Type `python run.py --helpfull` for more information.

## 3. Eval mode:
You can use `ben.py` for **eval** mode now. There are two ways to execute in eval mode:
1. Run on existing generated images v.s. test images: 
    ```bash
    python ben.py --input_dir=../Results/srcnn/SET14 --test=set14 --enable_psnr
    ```
    It will benchmark your just trained `srcnn`, print PSNR metric on `Set14` dataset.
    You can also use a reference dir to replace the dataset `--test=Set14`:
    ```bash
    python ben.py --input_dir=../Results/srcnn/SET14 --reference_dir=/datasets/set14/hr/ --enable_ssim
    ```
    Remember that the file order in reference dir must **exactly match** the order in input dir.

2. Run on model checkpoint:
    ```bash
    python ben.py --model=srcnn --epochs=50 --checkpoint_dir=../Results/srcnn/save --test=set5 --enable_psnr
    ```
    It will first load `srcnn` checkpoint and evaluate `Set14` data, then benchmark PSNR metric.
    Different from 1, this one saves disc space, and can run on an given epoch (as long as the checkpoint is saved, otherwise the latest checkpoint is loaded). 

#### Supported metrics in eval mode:
- PSNR (`--enable_psnr`)
- SSIM (`--enable_ssim`)
- Frechet Inception Distance (FID) (`--enable_fid`)
- Inception Score (`--enable_inception_score`)

#### Advanced options
- `--shave <uint>`: shave `<uint>` pixels from border before benchmark metrics.
- `--offset <int>`: if number of generated images is different from ground truth images,
                    offset `<int>` images during benchmarking. If `<int>` > 0, offset ground truth images,
                    if `<int>` < 0, offset generated images.
- `--l_only`: A flag, benchmark metrics only on luminance channel (A.K.A Y channel of YUV color space)
- `l_standard`: Use together with `--l_only` flag. This specify how to transform RGB image to YUV image.
                Please note different method will lead to different results. (The default value gives your best results :).)

#### Examples
1. VDSR
- Config: [vdsr.yaml](tf/vdsr.yaml)
    - overwrite model parameters:
        ```bash
        python run.py --model=vdsr --channel=3
        ```
        
        It will replace `vdsr: channel: 1` to `3` in runtime. Note that you can't overwrite parameter which is not declared in the config file.
- Train:
    
    `python run.py --model vdsr --epochs 100 --dataset 91-image --test none --infer ./test.png`
- Test:

    `python run.py --model vdsr --test set14`
- Infer:
    ```Bash
    python run.py --model vdsr --infer ./lr_image
    python run.py --model vdsr --infer ./mom.png
    ```
- Evaluate:
    ```Bash
    python ben.py --model=vdsr --checkpoint_dir=../Results/vdsr --epochs=100 --test=set14 --enable_psnr --enable_ssim
    python ben.py --input_dir=../Results/vdsr/SET14 --test=set14 --enable_psnr --enable_ssim
    ```

## 5. Dataset
Dataset is described in [dataset.yaml](../Data/datasets.yaml), see [README](../Data/README.md) for more details.
