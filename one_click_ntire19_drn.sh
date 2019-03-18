#!/usr/bin/env bash
# Author: Wenyi Tang
# Date: Mar. 18 2019
# Email: wenyitang@outlook.com

#DRN_TEST_MAT=
#DRN_SAVE_DIR=

# git clone https://github.com/LoSealL/VideoSuperResolution -b ntire_2019 && cd VideoSuperResolution

if [ ! -e setup.py ];
then
  echo " [!] Can't find setup.py file! Make sure you are in the right place!"
fi

echo "DRN_TEST_MAT=${DRN_TEST_MAT}"
echo "DRN_SAVE_DIR=${DRN_SAVE_DIR}"

pip install -e .
python prepare_data.py --filter=drn -q
echo " [*] Model extracted into Results/drn/save"
python VSR/Tools/DataProcessing/NTIRE19Denoise.py --validation=${DRN_TEST_MAT} --save_dir=${DRN_SAVE_DIR}/1/
pushd VSRTorch
python eval.py drn --cuda -t=../${DRN_SAVE_DIR}/1/ --output_index=0
popd
python VSR/Tools/DataProcessing/NTIRE19Denoise.py --results=Results/drn/1/ --save_dir=${DRN_SAVE_DIR}/2/
echo " [*] Processing done. Results are in ${DRN_SAVE_DIR}/2/"
