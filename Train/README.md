A one-line command to model executor
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
