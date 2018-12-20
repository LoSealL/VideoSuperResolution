"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 20th 2018

Prepare datasets and install VSR package for users.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import zipfile
import argparse
import sys
from pathlib import Path
from tensorflow.keras import utils as kutils

# For now VSR requires python>=3.5
if sys.version_info.major == 3 and sys.version_info.minor < 5:
    print("Python version is required >=3.5!")
    exit(-1)

_DEFAULT_DATASET_PATH = '/mnt/data/datasets'
_DEFAULT_DOWNLOAD_DIR = '/tmp/downloads'
_DEFAULT_WEIGHTS_DIR = './Results'
DATASETS = {
    'DIV2K': {
        'DIV2K_train_HR.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'DIV2K_valid_HR.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        'DIV2K_train_LR_unknown_X4.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip',
        'DIV2K_valid_LR_unknown_X4.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip',
    },
    'SET5.zip': 'https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip',
    'SET14.zip': 'https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip',
    'SunHay80.zip': 'https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip',
    'Urban100.zip': 'https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip',
    'VID4.zip': 'https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip',
    'BSD300.tgz': 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz',
    'BSD500.tgz': 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz',
    # '91image.rar': 'http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar',
    # 'waterloo.rar': 'http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar'
}
WEIGHTS = {
    'srcnn.tar.gz': 'https://github.com/LoSealL/Model/releases/download/srcnn_sc4_1/srcnn.tar.gz',
    'espcn.tar.gz': 'https://github.com/LoSealL/Model/releases/download/espcn_sc4_c1/espcn.tar.gz',
    'edsr.zip': 'https://github.com/LoSealL/Model/releases/download/edsr/edsr.zip',
}


def get_input(question):
    try:
        ans = input(question)
    except KeyboardInterrupt:
        ans = None
    return ans


def user_input(name, defaults=False):
    question = 'Do you wish to download {}? '.format(name)
    if defaults:
        question += '[Y/n] '
    else:
        question += '[y/N] '
    var = None
    while var is None:
        raw_ans = get_input(question)
        if raw_ans is None:
            print('\n', flush=True)  # user exit
            break
        elif raw_ans == '':
            var = defaults
            break
        ans = raw_ans.lower()
        if ans == 'y':
            var = True
        elif ans == 'n':
            var = False
        else:
            print('Invalid selection: {}'.format(raw_ans))
    return var


def download(name, url, path):
    fname = str(Path(path).resolve() / name)
    try:
        file = kutils.get_file(fname, url)
        return file
    except Exception:
        print('Unable to get file {}'.format(name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", type=str,
                        default=_DEFAULT_DOWNLOAD_DIR,
                        help="Specify download directory.")
    parser.add_argument("--data_dir", type=str,
                        default=_DEFAULT_DATASET_PATH,
                        help="Specify dataset extracted directoty.")
    parser.add_argument("--weights_dir", type=str,
                        default=_DEFAULT_WEIGHTS_DIR,
                        help="Specify weights extracted directory.")
    parser.add_argument("--yes_to_all", type=bool, default=False)
    args = parser.parse_args()
    # make work dir
    Path(args.download_dir).mkdir(exist_ok=True, parents=True)
    Path(args.data_dir).mkdir(exist_ok=True, parents=True)

    def get_leaf(key: str, node: dict):
        for k, v in node.items():
            if isinstance(v, dict):
                for k2, v2 in get_leaf(k, v):
                    yield Path(key) / k2, v2
            else:
                yield Path(key) / k, v

    need_to_download = {}
    for k, v in get_leaf(args.data_dir, DATASETS):
        if user_input(k.stem, args.yes_to_all):
            need_to_download[k] = v
    for k, v in get_leaf(args.weights_dir, WEIGHTS):
        if user_input(k.stem, args.yes_to_all):
            need_to_download[k] = v
    need_to_extract = {}
    for k, v in need_to_download.items():
        need_to_extract[k] = (k.parent, download(k.name, v, args.download_dir))
    for k, v in need_to_extract.values():
        if v is None:
            continue
        ext = Path(v).suffix
        if ext in ('.tar', '.tgz', '.gz', '.bz'):
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        elif ext in ('.zip',):
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        if is_match_fn(v):
            with open_fn(v) as fd:
                try:
                    fd.extractall(str(k.resolve()))
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    # TBD...
                    pass


if __name__ == '__main__':
    main()
