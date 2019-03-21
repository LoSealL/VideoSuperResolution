## One-click script
Script codes in https://github.com/LoSealL/NTIRE2019-Competition-Test

You should also have `tensorflow(-gpu)>=1.12.0` and `pytorch>=1.0.0` installed manually.

Before running `one_click_ntire19_rsr.sh` for Real Image Super Resolution, set two paths: `RSR_TEST_DIR` for testing images and `RSR_SAVE_DIR` for saving results.
```bash
RSR_TEST_DIR=bla/bla/bla
RSR_SAVE_DIR=bli/bli/bli
. one_click_ntire19_rsr.sh
```
 
Before running `one_click_ntire19_drn.sh` for sRGB Image Denoising, set two paths: `DRN_TEST_MAT` for testing mat file and `DRN_SAVE_DIR` for saving results.
```bash
DRN_TEST_MAT=bla/bla/bla/BenchmarkNoisyBlocksSrgb.mat
DRN_SAVE_DIR=bli/bli/bli
. one_click_ntire19_drn.sh
```

You can also do it step-by-step as follows.

## Step by step reproduce instructions

1. Install the whole VSR package and its requirements:
    ```bash
    git clone https://github.com/LoSealL/VideoSuperResolution -b ntire_2019 && cd VideoSuperResolution
    pip install -e .
    ```
    Note that you should pre-install `tensorflow` and `pytorch`.

2. Download the pre-trained model:
   
   **make sure you are in the root folder.*
   
   For Real Image Super-Resolution
   ```bash
   python prepare_data.py --filter=rsr -q
   ```
   
   For sRGB Real Image Denoising (Track #2: sRGB)
   ```bash
   python prepare_data.py --filter=drn -q
   ```

   Model url for manually download:
   - [rsr](https://github.com/LoSealL/Model/releases/download/crdn/rsr.zip): https://github.com/LoSealL/Model/releases/download/crdn/rsr.zip
   - [drn](https://github.com/LoSealL/Model/releases/download/mldn/drn.zip): https://github.com/LoSealL/Model/releases/download/mldn/drn.zip
   
3. Prepare testing data:

   **make sure you are in the root folder.*
   
   For RSR:
   
   You need to crop images into small patches by:
   ```bash
   python VSR/Tools/DataProcessing/NTIRE19RSR.py --ref_dir=path/to/test/data/folder --patch_size=768 --stride=760 --save_dir=path/to/saving/folder
   ```
   
   For sRGB Denoising:
   
   You need to convert .MAT file to png images by:
   ```bash
   python VSR/Tools/DataProcessing/NTIRE19Denoise.py --validation=path/to/.MAT --save_dir=path/to/saving/folder
   ```
   
4. Predicting

   **make sure you are in the root folder.*
   
   For RSR:
   Entering VSRTorch folder
   ```bash
   cd VSRTorch
   python eval.py rsr --cuda -t=/path/to/divided/test/images/folder --pth=../Results/rsr/save/rsr_ep2000.pth --ensemble
   ```
   The output will be saved in `../Results/rsr/<your-image-folder-name>`. To combine them back together:
   ```bash
   cd ..
   python VSR/Tools/DataProcessing/NTIRE19RSR.py --ref_dir=path/to/test/data/folder --patch_size=768 --stride=760 --results=Results/rsr/<your-image-folder>/ --save_dir=path/to/saving/folder
   ```
   Where `--ref_dir` should keep the same as the folder in step 3, it's a reference to know how to combine patches. `--patch_size` and `--stride` should also keep the same.
   
   For sRGB Denoising:
   Entering VSRTorch folder
   ```bash
   cd VSRTorch
   python eval.py drn --cuda -t=/path/to/divided/test/images/folder --pth=../Results/drn/save/drn_ep2000.pth --output_index=0 --ensemble
   ```
   The output will be saved in `../Results/drn/<your-image-folder-name>`. To pack them into mat file:
   ```bash
   cd ..
   python VSR/Tools/DataProcessing/NTIRE19Denoise.py --results=Results/drn/<your-image-folder-name> --save_dir=path/to/saving/folder
   ```
   
   *If OOM happened, try not to enable `--cuda` flag.