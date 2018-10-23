"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Sep 13th 2018

Image customized decoder for YV12([Y][U/4][V/4]), YV21([Y][V/4][U/4])
NOTE: [Y][U][V] means Y/U/V channel is a planar channel, [U/4] means
  U channel is sub-sampled by a factor of [2, 2]
"""
from PIL import Image, ImageFile
import numpy as np


class YV12Decoder(ImageFile.PyDecoder):
    """PIL.Image.DECODERS for YV12 format raw bytes

    Registered in `Image.DECODERS`, don't use this class directly!
    """

    def __init__(self, mode, *args):
        super(YV12Decoder, self).__init__(mode, *args)

    def decode(self, buffer):
        if self.mode == 'L':
            # discard UV channel
            self.set_as_raw(buffer, 'L')
        else:
            W, H = self.im.size
            y = np.frombuffer(buffer, 'uint8', count=W * H)
            u = np.frombuffer(buffer, 'uint8', count=W * H // 4, offset=W * H)
            v = np.frombuffer(buffer, 'uint8', count=W * H // 4, offset=W * H + W * H // 4)
            y = np.reshape(y, [H, W])
            u = np.reshape(u, [H // 2, W // 2])
            v = np.reshape(v, [H // 2, W // 2])
            u = u[np.arange(H) // 2][:, np.arange(W) // 2]
            v = v[np.arange(H) // 2][:, np.arange(W) // 2]
            yuv = np.stack([y, u, v], axis=-1)
            self.set_as_raw(yuv.flatten().tobytes())
        return -1, 0


class YV21Decoder(ImageFile.PyDecoder):
    """PIL.Image.DECODERS for YV21 format raw bytes

    Registered in `Image.DECODERS`, don't use this class directly!
    """

    def __init__(self, mode, *args):
        super(YV21Decoder, self).__init__(mode, *args)

    def decode(self, buffer):
        if self.mode == 'L':
            # discard UV channel
            self.set_as_raw(buffer, 'L')
        else:
            W, H = self.im.size
            y = np.frombuffer(buffer, 'uint8', count=W * H)
            v = np.frombuffer(buffer, 'uint8', count=W * H // 4, offset=W * H)
            u = np.frombuffer(buffer, 'uint8', count=W * H // 4, offset=W * H + W * H // 4)
            y = np.reshape(y, [H, W])
            u = np.reshape(u, [H // 2, W // 2])
            v = np.reshape(v, [H // 2, W // 2])
            u = u[np.arange(H) // 2][:, np.arange(W) // 2]
            v = v[np.arange(H) // 2][:, np.arange(W) // 2]
            yuv = np.stack([y, u, v], axis=-1)
            self.set_as_raw(yuv.flatten().tobytes())
        return -1, 0


Image.DECODERS['YV12'] = YV12Decoder
Image.DECODERS['YV21'] = YV21Decoder
