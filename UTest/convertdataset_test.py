import os

if not os.getcwd().endswith('UTest'):
  os.chdir('UTest')
from VSR.Tools.DataProcessing import ConvertDatasets
from PIL import Image
import numpy as np


def test_image_process():
  url = "./data/set5_x2/img_001_SRF_2_LR.png"
  crf = np.load("./data/crf_s.npz")
  image = np.asarray(Image.open(url))
  proc = ConvertDatasets.process(image, crf, (0.16, 0.06))
  assert abs(proc.mean() - image.mean()) < 10
  # Image.fromarray(proc, 'RGB').show()
