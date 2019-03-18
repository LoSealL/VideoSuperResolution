#!/usr/bin/env bash

#DRN_TEST_MAT=
#DRN_SAVE_DIR=
_PATCH_SIZE=768
_STRIDE=760

if [ ! -e setup.py ];
then
  echo " [!] Can't find setup.py file! Make sure you are in the right place!"
fi

echo "DRN_TEST_MAT=${DRN_TEST_MAT}"
echo "DRN_SAVE_DIR=${DRN_SAVE_DIR}"

pip install -e .
python prepare_data.py --filter=drn
echo " [*] Model extracted into Results/drn/save"
python VSR/Tools/DataProcessing/NTIRE19Denoise.py --validation=${DRN_TEST_MAT} --save_dir=${DRN_SAVE_DIR}/1/
pushd VSRTorch
python eval.py drn --cuda -t=${DRN_SAVE_DIR}/1/
popd
python VSR/Tools/DataProcessing/NTIRE19Denoise.py --results=Results/drn/1/ --save_dir=${DRN_SAVE_DIR}/2/
echo " [*] Processing done. Results are in ${DRN_SAVE_DIR}/2/"
