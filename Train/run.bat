python train.py edsr ^
  --retrain=0 --scale=4 --epochs=500 --dataset=DIV2K ^
  --channel=3 --batch=16 --patch_size=192 --random_patches=200 ^
  --custom_feature_cb="" ^
  --comment=baseline
