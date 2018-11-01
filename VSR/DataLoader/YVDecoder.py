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
            w, h = self.im.size
            y = np.frombuffer(buffer, 'uint8', count=w * h)
            u = np.frombuffer(buffer, 'uint8', count=w * h // 4, offset=w * h)
            v = np.frombuffer(
                buffer, 'uint8', count=w * h // 4, offset=w * h + w * h // 4)
            y = np.reshape(y, [h, w])
            u = np.reshape(u, [h // 2, w // 2])
            v = np.reshape(v, [h // 2, w // 2])
            u = u[np.arange(h) // 2][:, np.arange(w) // 2]
            v = v[np.arange(h) // 2][:, np.arange(w) // 2]
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
            w, h = self.im.size
            y = np.frombuffer(buffer, 'uint8', count=w * h)
            v = np.frombuffer(buffer, 'uint8', count=w * h // 4, offset=w * h)
            u = np.frombuffer(
                buffer, 'uint8', count=w * h // 4, offset=w * h + w * h // 4)
            y = np.reshape(y, [h, w])
            u = np.reshape(u, [h // 2, w // 2])
            v = np.reshape(v, [h // 2, w // 2])
            u = u[np.arange(h) // 2][:, np.arange(w) // 2]
            v = v[np.arange(h) // 2][:, np.arange(w) // 2]
            yuv = np.stack([y, u, v], axis=-1)
            self.set_as_raw(yuv.flatten().tobytes())
        return -1, 0


Image.DECODERS['YV12'] = YV12Decoder
Image.DECODERS['YV21'] = YV21Decoder
