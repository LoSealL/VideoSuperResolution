"""
# Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.

Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 21st 2018

Pre-processing DND dataset:
- Crop each image into 20 small patches (1000 in total)
- Convert .mat into png files
- Submit denoised .png files into bundled .mat files.
"""

import numpy as np
import scipy.io as sio
import os
import h5py
import argparse
from PIL import Image
from pathlib import Path


def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf


def load_sigma_raw(info, img_id, bb, yy, xx):
    nlf_h5 = info[info["sigma_raw"][0][img_id]]
    sigma = nlf_h5[xx, yy, bb]
    return sigma


def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0, bb]
    return sigma


def denoise_raw(denoiser, data_folder, out_folder):
    '''
    Utility function for denoising all bounding boxes in all raw images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch
                  and nlf is a dictionary containing the parameters of the noise level
                  function (nlf["a"], nlf["b"]) and a mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:
        pass

    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_raw', '%04d.mat' % (i + 1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['Inoisy']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1),
                   int(boxes[k, 3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3]].copy()
            Idenoised_crop = Inoisy_crop.copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            nlf = load_nlf(info, i)
            for yy in range(2):
                for xx in range(2):
                    nlf["sigma"] = load_sigma_raw(info, i, k, yy, xx)
                    Inoisy_crop_c = Inoisy_crop[yy:H:2, xx:W:2].copy()
                    Idenoised_crop_c = denoiser(Inoisy_crop_c, nlf)
                    Idenoised_crop[yy:H:2, xx:W:2] = Idenoised_crop_c
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)
            save_file = os.path.join(out_folder,
                                     '%04d_%02d.mat' % (i + 1, k + 1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k + 1, 20))
        print('[%d/%d] %s done\n' % (i + 1, 50, filename))


def denoise_srgb(denoiser, data_folder, out_folder):
    '''
    Utility function for denoising all bounding boxes in all sRGB images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch
                  and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:
        pass

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_srgb',
                                '%04d.mat' % (i + 1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1),
                   int(boxes[k, 3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            nlf = load_nlf(info, i)
            for yy in range(2):
                for xx in range(2):
                    nlf["sigma"] = load_sigma_srgb(info, i, k)
                    Idenoised_crop = denoiser(Inoisy_crop, nlf, index=(i, k))
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)
            save_file = os.path.join(out_folder,
                                     '%04d_%02d.mat' % (i + 1, k + 1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k + 1, 20))
        print('[%d/%d] %s done\n' % (i + 1, 50, filename))


def bundle_submissions_raw(submission_folder):
    '''
    Bundles submission data for raw denoising

    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''

    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:
        pass

    israw = True
    eval_version = "1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%02d.mat' % (i + 1, bb + 1)
            s = sio.loadmat(os.path.join(submission_folder, filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat' % (i + 1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )


def bundle_submissions_srgb(submission_folder):
    '''
    Bundles submission data for sRGB denoising

    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:
        pass
    israw = False
    eval_version = "1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%02d.mat' % (i + 1, bb + 1)
            s = sio.loadmat(os.path.join(submission_folder, filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat' % (i + 1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="dnd data folder (root folder).")
    parser.add_argument("--save_dir", type=str, default='./outputs',
                        help="output cropped patches.")
    parser.add_argument("--submission_dir", type=str, default=None,
                        help="denoised images (.png).")
    parser.add_argument("--bundle_dir", type=str, default='./bundled')
    args = parser.parse_args()


    def denoiser(image, nlf, index):
        i, k = index
        name = '%04d_%02d.png' % (i + 1, k + 1)
        image_u = np.round(image * 255).astype(np.uint8)
        image_u = Image.fromarray(image_u, 'RGB')
        image_u.save(os.path.join(args.save_dir, name))
        name = '%04d_%02d_lr.png' % (i + 1, k + 1)
        image_u.resize([image_u.width // 4, image_u.height // 4],
                       Image.BICUBIC).save(os.path.join(args.save_dir, name))
        return image


    if args.data_dir:
        denoise_srgb(denoiser, args.data_dir, args.save_dir)
    if args.submission_dir:
        for img_file in Path(args.submission_dir).glob('*.png'):
            img_u = Image.open(img_file)
            img = np.asarray(img_u, np.float32) / 255
            path = Path(args.bundle_dir) / (img_file.stem[:7] + '.mat')
            sio.savemat(str(path), {'Idenoised_crop': img})
        bundle_submissions_srgb(args.bundle_dir)
