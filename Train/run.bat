python train.py exp2 ^
  --retrain=0 --scale=1 --epochs=5 --dataset=DIV2K ^
  --channel=3 --batch=16 --patch_size=96 --random_patches=200 ^
  --custom_feature_cb=noisy ^
  --comment=debug
