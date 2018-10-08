from VSR.Util.ImageProcess import *

IMG = 'data/set5_x2/img_001_SRF_2_LR.png'
img = imread(IMG)
img = img.astype('float32')

yuv = rgb_to_yuv(img, 255, 'matlab')
array_to_img(yuv).show()
