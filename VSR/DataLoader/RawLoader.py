import numpy as np

from pathlib import Path
from io import SEEK_END
from time import time

from DataLoader.Dataset import Dataset
from Util import ImageProcess


class RawLoader(object):
    def __init__(self, mode='YV12', dataset_name='yang-91', verbose=True):
        _allowed_mode = [
            'YV12',
            'YV21',
            'NV12',
            'NV21',
            'RGB4',
            'BGR4'
        ]
        if not mode.upper() in _allowed_mode:
            raise ValueError(f'unknown mode: {mode}')
        if not dataset_name.upper() in Dataset.DATASET.keys():
            raise ValueError(f'unknown dataset: {dataset_name}')
        self.mode = mode
        self.dataset = Dataset.DATASET[dataset_name.upper()]
        self.pitch, self.channel_pitch = self._get_frame_pitch(mode, [self.dataset.height, self.dataset.width])
        self.shape = self._get_frame_channel_shape(mode, [self.dataset.height, self.dataset.width])
        self.bucket = []
        self.built = False
        self.verbose = verbose

    def _estimate(self, method='train'):
        _allowed_mathod = ['train', 'eval', 'val', 'test']
        if not method.lower() in _allowed_mathod:
            raise ValueError(f'unknown method: {method}')
        if method.lower() == 'eval':
            method = 'val'
        frame_no = dict()
        for path in self.dataset.__getattr__(method.lower()):
            file = Path(path).open('rb')
            file.seek(0, SEEK_END)
            file_size = file.tell()
            frame_no[Path(path).stem] = (file_size // self.pitch)
            assert file_size % self.pitch == 0
            file.close()
        return frame_no

    def _crop_index(self, stride, size, stack):
        if isinstance(size, int):
            size = (size, size)
        if isinstance(size, (tuple, list)):
            size = np.asarray(size, np.int32)
        if not isinstance(size, np.ndarray) or size.ndim != 1:
            raise TypeError('invalid length type, must be a 2-D tensor or an int')
        w = self.dataset.width
        h = self.dataset.height
        axis0 = list(range(0, w, stride))
        axis1 = list(range(0, h, stride))
        if w - axis0[-1] < size[0]:
            axis0 = axis0[:-1]
        if h - axis1[-1] < size[1]:
            axis1 = axis1[:-1]
        crop_index = []
        for i in range(stack):
            for x in axis0:
                for y in axis1:
                    crop_index.append(np.array([i, x, y, x + size[0], y + size[1]], dtype=np.int32))
        return crop_index

    @staticmethod
    def _get_frame_pitch(mode, size):
        """Get bytes length of one frame.
        For the detail of mode fourcc, please see https://www.fourcc.org/

        :param mode: Must be any of {YV12, YV21, NV12, NV21, RGB, BGR}
        :param size: frame length, must be a tuple or list
        :return: frame bytes, [channel0 bytes, channel1 bytes, channel2 bytes, ...]

        RGB, BGR, and UV channel of NV12, NV21 is packed, while YV12 and YV21 is planar, hence we have:
          - **channel0** of YV12, YV21, NV12, NV21 if Y
          - **channel1** of YV12 is U, of YV21 is V, of NV12 is UV, of NV21 is VU
          - **channel2** of YV12 is V, of YV21 is U
        """

        if mode in ('YV12', 'YV21'):
            return size[0] * size[1] * 3 // 2, [size[0] * size[1], size[0] * size[1] // 4, size[0] * size[1] // 4]
        if mode in ('NV12', 'NV21'):
            return size[0] * size[1] * 3 // 2, [size[0] * size[1], size[0] * size[1] // 2]
        if mode in ('RGB', 'BGR'):
            return size[0] * size[1] * 3, [size[0] * size[1] * 3]

    @staticmethod
    def _get_frame_channel_shape(mode, size):
        """Get each channel's shape according to mode and frame length.
        For the detail of mode fourcc, please see https://www.fourcc.org/

        :param mode: Must be any of {YV12, YV21, NV12, NV21, RGB, BGR}
        :param size: frame length, must be a tuple or list
        :return: 3-D tensor of [C, W, H] for YV12, YV21, NV12, NV21; [W, H, C] for RGB and BGR
        """
        if mode in ('YV12', 'YV21'):
            return np.array([1, size[0], size[1]]), np.array([1, size[0] // 2, size[1] // 2]), np.array(
                [1, size[0] // 2, size[1] // 2])
        if mode in ('NV12', 'NV21'):
            return np.array([1, size[0], size[1]]), np.array([2, size[0] // 2, size[1] // 2])
        if mode in ('RGB', 'BGR'):
            return np.array([size[0], size[1], 3])

    def build(self, method, seq_depth):
        frames = self._estimate(method)
        total_frames = sum(frames.values())
        if self.verbose:
            print(f"Building {total_frames} frames from {method} set")
        start_time_0 = time()
        for path in self.dataset.__getattr__(method):
            start_time = time()
            with Path(path).open('rb') as fd:
                data = np.fromfile(fd, 'uint8')
                data = np.split(data, range(self.pitch, data.size, self.pitch), axis=0)
                data = data[:len(data) - len(data) % seq_depth]
                channel_split = [sum(self.channel_pitch[:i]) for i in range(1, len(self.channel_pitch))]
                for i in range(0, len(data), seq_depth):
                    yuv_seq = []
                    for frame in data[i:i+seq_depth]:
                        frame = np.split(frame, channel_split)
                        channel = [np.reshape(frame[i], self.shape[i]) for i in range(len(frame))]
                        yuv_seq.append(ImageProcess.img_to_yuv(channel, self.mode, grayscale=True))
                    # yuv_seq_lr = [image.bicubic_rescale(img, scale) for img in yuv_seq]
                    self.bucket.append(tuple(yuv_seq))
            delta_time = time() - start_time
            if self.verbose:
                fps = frames[Path(path).stem] / delta_time
                total_frames -= frames[Path(path).stem]
                eta = total_frames / fps
                print(f'Building fps: {fps:.2f}   Remain {total_frames}   ETA: {eta:.2f}s', end='        \r')
        self.built = True
        if self.verbose:
            print(f'\nBuild done! Time collapsed: {time() - start_time_0:.2f}s')

    def load_batch(self, batch=1, scale=3, patch_size=48, stride=1, shuffle=False):
        if not self.built:
            raise RuntimeError('this loader has not been built! Call self.build first.')
        patch_size = patch_size - patch_size % scale
        crop_index = self._crop_index(stride, patch_size, len(self.bucket))
        while True:
            if shuffle:
                np.random.shuffle(crop_index)
            for i in range(0, len(crop_index), batch):
                batch_seq_hr = []
                batch_seq_lr = []
                for index in crop_index[i:i+batch]:
                    batch_hr = [seq.crop(index[1:]) for seq in self.bucket[index[0]]]
                    batch_lr = [ImageProcess.bicubic_rescale(img, 1 / scale) for img in batch_hr]
                    batch_seq_hr.append(np.stack([ImageProcess.img_to_array(img) for img in batch_hr]))
                    batch_seq_lr.append(np.stack([ImageProcess.img_to_array(img) for img in batch_lr]))
                yield np.stack(batch_seq_lr), np.stack(batch_seq_hr)  # [B, S, H, W, C]

    def debug_info_print(self):
        try:
            frame_no = self._estimate(method='train')
            print(frame_no)
            frame_no = self._estimate(method='val')
            print(frame_no)
            frame_no = self._estimate(method='test')
            print(frame_no)
        except ValueError or TypeError as ex:
            print(f'Exception: {ex}')
