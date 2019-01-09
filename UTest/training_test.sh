#!/usr/bin/env bash
pushd ./Train

python run.py --model=srcnn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --test=bar
python run.py --mode=eval --model=srcnn --checkpoint_dir=../Results/srcnn --test=bar --enable_psnr
python run.py --model=espcn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --test=bar
python run.py --model=vdsr --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --test=bar
python run.py --model=drcn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --test=bar
python run.py --model=drrn --data_config=../UTest/data/fake_datasets.yml --dataset=normal --epochs=1 --test=bar
popd
