#!/usr/bin/env bash
# show model configuration
cat tf/edsr.yaml

# Train selective model (checkpoint saved under ../Results/edsr/save)
python run.py --model=edsr --dataset=div2k --threads=16 --epochs=200

# Evaluation selective test dataset (tests saved under ../Results/edsr/SET5)
python run.py --model=edsr --test=set5

# Make inference on custom images (i.e. given images under ./lr_images/*.png)
# (Outputs saved under ../Results/edsr/infer)
python run.py --model=edsr --infer=./lr_images
python run.py --model=edsr --infer=./lr_images/001.png

# Show benchmarks (results saved under /tmp/vsr/)
python run.py --mode=eval --model=edsr --checkpoint_dir=../Results/edsr --epochs=200 --test=set5 --enable_psnr --enable_ssim
python run.py --mode=eval --input_dir=../Results/edsr/SET5 --test=set5 --enable_psnr --enable_ssim
