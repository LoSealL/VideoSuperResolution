#!/usr/bin/env bash
# Author: Wenyi Tang
# Date: Mar. 18 2019
# Email: wenyitang@outlook.com

#RSR_TEST_DIR=
#RSR_SAVE_DIR=
_PATCH_SIZE=768
_STRIDE=760

# git clone https://github.com/LoSealL/VideoSuperResolution -b ntire_2019 && cd VideoSuperResolution

if [ ! -e setup.py ];
then
  echo " [!] Can't find setup.py file! Make sure you are in the right place!"
fi

echo "RSR_TEST_DIR=${RSR_TEST_DIR}"
echo "RSR_SAVE_DIR=${RSR_SAVE_DIR}"

pip install -e .
python prepare_data.py --filter=rsr -q
echo " [*] Model extracted into Results/rsr/save"
python VSR/Tools/DataProcessing/NTIRE19RSR.py --ref_dir=${RSR_TEST_DIR} --patch_size=${_PATCH_SIZE} --stride=${_STRIDE} --save_dir=${RSR_SAVE_DIR}/1/
pushd VSRTorch
python eval.py rsr --cuda --ensemble -t=../${RSR_SAVE_DIR}/1/
popd
python VSR/Tools/DataProcessing/NTIRE19RSR.py --ref_dir=${RSR_TEST_DIR} --patch_size=${_PATCH_SIZE} --stride=${_STRIDE} --results=Results/rsr/1/ --save_dir=${RSR_SAVE_DIR}/2/
echo " [*] Processing done. Results are in ${RSR_SAVE_DIR}/2/"
