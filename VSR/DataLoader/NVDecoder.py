"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Sep 13th 2018

Image customized decoder for NV12([Y][UV/4]), NV21([Y][VU/4])
NOTE: [Y] means Y channel is a planar channel, [UV] means UV
  channels together is planar, but U and V are packed. [UV/4]
  means U and V are sub-sampled by a factor of [2, 2]
"""
from PIL import Image, ImageFile
import numpy as np


class NV12Decoder(ImageFile.PyDecoder):
    """PIL.Image.DECODERS for NV12 format raw bytes

    Registered in `Image.DECODERS`, don't use this class directly!
    """

    def __init__(self, mode, *args):
        super(NV12Decoder, self).__init__(mode, *args)

    def decode(self, buffer):
        if self.mode == 'L':
            # discard UV channel
            self.set_as_raw(buffer, 'L')
        else:
            W, H = self.im.size
            y = np.frombuffer(buffer, 'uint8', count=W * H)
            uv = np.frombuffer(buffer, 'uint8', count=W * H // 2, offset=W * H)
            y = np.reshape(y, [H, W, 1])
            uv = np.reshape(uv, [H // 2, W // 2, 2])
            uv = uv[np.arange(H) // 2][:, np.arange(W) // 2]
            yuv = np.concatenate([y, uv], axis=-1)
            self.set_as_raw(yuv.flatten().tobytes())
        return -1, 0


class NV21Decoder(ImageFile.PyDecoder):
    """PIL.Image.DECODERS for NV21 format raw bytes

    Registered in `Image.DECODERS`, don't use this class directly!
    """

    def __init__(self, mode, *args):
        super(NV21Decoder, self).__init__(mode, *args)

    def decode(self, buffer):
        if self.mode == 'L':
            # discard UV channel
            self.set_as_raw(buffer, 'L')
        else:
            W, H = self.im.size
            y = np.frombuffer(buffer, 'uint8', count=W * H)
            vu = np.frombuffer(buffer, 'uint8', count=W * H // 2, offset=W * H)
            y = np.reshape(y, [H, W, 1])
            vu = np.reshape(vu, [H // 2, W // 2, 2])
            vu = vu[np.arange(H) // 2][:, np.arange(W) // 2]
            uv = vu[:, :, ::-1]
            yuv = np.concatenate([y, uv], axis=-1)
            self.set_as_raw(yuv.flatten().tobytes())
        return -1, 0


Image.DECODERS['NV12'] = NV12Decoder
Image.DECODERS['NV21'] = NV21Decoder
