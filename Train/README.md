A one-line command to model executor
## New: eval mode
Now you can run an **eval** mode in `run.py`.

```bash
python run.py --mode=eval --model=srcnn --epochs=100 --dataset=set5 --checkpoint_dir=/tmp/srcnn/save \
  --enable_psnr --enable_ssim --l_only
```
The above code will run srcnn model (`--model=srcnn`) using checkpoint 
`*ep0100.ckpt` (`--epochs=100`) to inference **SET5** images (`--dataset=set5` or `--test=set5`)
and calculate PSNR, SSIM metrics using only luminance channel (`--l_only`).

```bash
python run.py --mode=eval --input_dir=/tmp/set5_sr --dataset=set5 --enable_psnr
```
The above code will calculate PSNR metric of SET5 and images in `/tmp/set5_sr`

(_Note_: you have to make sure `/tmp/set5_sr` has exactly 5 images with the same
shape and the same loading order as Set5, but can have different file name. I.e.
`img_001_SRF_4_HR.img` v.s. `img_001_SRF_4_SR.img`)

### Supported metrics:
- PSNR
- SSIM
- Frechet Inception Distance (FID)
- Inception Score

## How to use
`run.py` can be used to train/benchmark/infer models collected in [VSR.Models](../VSR/Models)

The model parameters can be configured through [parameters/root.yaml](./parameters/root.yaml) and
*<parameters/model-name.yaml>* (i.e. [srcnn](./parameters/srcnn.yaml))

```Bash
python run.py --model <model-name> --epochs <num> --steps_per_epoch <num> --dataset <train-dataset-name> \
  --test <test-dataset-name> --infer <infer-dir | single-file | infer-dataset-name> --threads <loading-thread-num>\
  --save_dir <save>
```

Type `python run.py --help` for more information

## Examples
1. VDSR
- Config: [vdsr.yaml](parameters/vdsr.yaml)
- Train:
    
    `python run.py --model vdsr --epochs 100 --dataset 91-image --test none --infer ./test.png`
- Test:

    `python run.py --model vdsr --test set14`
- Infer:
    ```
    python run.py --model vdsr --infer ./lr_image
    python run.py --model vdsr --infer ./mom.png
    ```

## Run Benchmark
Calculate PSNR and SSIM for Set5 outputs and labels, exclude 4-pixel boarder:

    `python metrics.py --dataset=set5 --input_dir=./Outputs/set5 --shave=4`

Don't calculate SSIM:
    
    `python metrics.py --dataset=set5 --input_dir=./Outputs/set5 --shave=4 --no_ssim`

Calculate PSNR for video set VID4:

    `python metrics.py --dataset=vid4 --input_dir=./Outputs/vid4`


## Dataset
Dataset is described in [dataset.yaml](../Data/datasets.yaml), see [README](../Data/README.md) for more details.
