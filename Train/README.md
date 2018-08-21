A one-line command to train models
## Usage
- `python train.py srcnn --batch=128`
- `python train.py rdn --epochs=100 --dataconfig=../Data/datasets.json --dataset=BSD`

Please check [model_alias.py](./model_alias.py) to find out supported models.

## NOTE
This is easy to use, however if you want to fine-tune your model, please consider writing your own script by trying VSR.

## Run Benchmark
Calculate PSNR and SSIM for Set5 outputs and labels, exclude 4-pixel boarder:
- `python metrics --dataset=set5 --input_dir=./Outputs/set5 --shave=4`

Don't calculate SSIM:
- `python metrics --dataset=set5 --input_dir=./Outputs/set5 --shave=4 --no_ssim`

Calculate PSNR for video set VID4:
- `python metrics --dataset=vid4 --input_dir=./Outputs/vid4`

In folder `./Outputs/vid4`, there are 4 sub-folders: calender, city, foliage, walk.
Each contains png frames. 

In `datasets.json`:

```json
{
  "Path_Tracked": {
    "SET5": <path-to-set5>,
    "VID4": <path-to-vid4>
  }
}
```

Folder strucure:
```
-Set5
|- 001.png
|- 002.png
|- 003.png
|- 004.png
|- 005.png
-Vid4
|- calender
|--|-- 001.png
|--|-- ...
|- city
|--|-- ...
|- foliage
|--|-- ...
|- walk
|--|-- ...
```