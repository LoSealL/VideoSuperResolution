#!/usr/bin/env bash
pushd ./Train
# Test SISR Models
python run.py --model=srcnn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=espcn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=vdsr --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=drcn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=drrn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=carn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=dbpn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=dcscn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=edsr --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=idn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=lapsrn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=memnet --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=msrn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=rcan --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=rdn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=srdensenet --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=srfeat --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=srgan --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
python run.py --model=dncnn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --steps_per_epoch=1
# Test VSR Models

# Test GAN Models
python run.py --model=sgan --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=lsgan --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=rgan --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=ragan --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=rlsgan --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=ralsgan --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=wgan --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=gangp --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=wgangp --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=rgangp --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
python run.py --model=ragangp --data_config=../UTest/data/fake_datasets.yml --dataset=numpy --epochs=1 --steps_per_epoch=1
# Test eval mode
python run.py --mode=eval --model=srcnn --checkpoint_dir=../Results/srcnn --test=bar --enable_psnr --enable_ssim --steps_per_epoch=1 --data_config=../UTest/data/fake_datasets.yml
popd
