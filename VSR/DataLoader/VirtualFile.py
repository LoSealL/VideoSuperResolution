"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 9th 2018
Updated Date: May 9th 2018

virtual file is an abstraction of a file or
a collection of ordered files
"""
from pathlib import Path
from io import SEEK_END, BytesIO
from PIL import Image
import numpy as np

from ..Util.Utility import to_list


class File:

    def __init__(self, path, rewind=False):
        """
        If path is a file, File opens it and calculates its length.
        If path is a folder, File organize each file in the folder as alphabet order

        Args:
             path: path to a node (can be a file or just a folder)
             rewind: rewind the file when reaches EOF
        """
        self.path = Path(path)
        self.file = []
        self.length = dict()
        mode = 'rb'  # mode must be 'rb'
        if self.path.is_file():
            self.file = [self.path]
            with self.path.open(mode) as fd:
                fd.seek(0, SEEK_END)
                self.length[self.path.name] = fd.tell()
        elif self.path.is_dir():
            for _file in self.path.glob('*'):
                self.file.append(_file)
                with _file.open(mode) as fd:
                    fd.seek(0, SEEK_END)
                    self.length[_file.name] = fd.tell()
        self.read_file = []
        self.read_pointer = 0
        self.end_pointer = sum(self.length.values())
        self.cur_fd = None
        self.rewind = rewind

    def read(self, count=None):
        """
        Read `count` bytes

        Args:
            count: number of bytes to be read, if None, read all bytes of **1** file

        Return:
            bytes read
        """
        if count == 0:
            return b''
        if not self.cur_fd and self.file:
            self.cur_fd = self.file[0].open('rb')
            self.read_file.append(self.file[0])
            self.file.pop(0)
        elif not self.cur_fd:
            raise FileNotFoundError('No files in File')
        read_bytes = self.cur_fd.read(count)
        if read_bytes:
            self.read_pointer += len(read_bytes)
            if count and count > len(read_bytes):
                return read_bytes + self.read(count - len(read_bytes))
            else:
                return read_bytes
        else:
            if self.file:
                self.cur_fd.close()
                self.cur_fd = self.file[0].open('rb')
                self.read_file.append(self.file[0])
                self.file.pop(0)
                return self.read(count)
            elif self.rewind and self.read_file:
                self.file = self.read_file.copy()
                self.read_file.clear()
                self.cur_fd = None
                return self.read(count)
            else:
                raise EOFError('End of File!')

    def read_frame(self, frames=1, *args):
        pass

    def seek(self, offset, where):
        """
        Seek the position by `offset` relative to `where`

        Args:
             offset: move the read pointer by `offset` bytes
             where: could be io.SEEK_END, io.SEEK_CUR, io.SEEK_SET
        """
        pass

    def tell(self):
        """
        Tell the current position of the read pointer
        """
        return self.read_pointer

    def __len__(self):
        return self.end_pointer

    def size(self, name):
        """
        Get the length of the file named `name`

        Return:
            length in bytes
        """
        path = Path(name)
        name = path.stem if path.exists() else name
        return self.length.get(name)


class RawFile(File):

    def __init__(self, path, mode, size, rewind=False):
        """
        Initiate Raw object. The file is lazy loaded, which means
        the file is opened but not loaded into memory.

        Arguments:
             path: file path or handle
             mode: since raw file has no headers, type must be explicitly given
             size: a tuple of (width, height), must be explicitly given
             rewind: rewind the file when reaches EOF

        Raise:
            TypeError
        """

        _allowed_mode = [
            'YV12',
            'YV21',
            'NV12',
            'NV21',
            'RGB4',
            'BGR4'
        ]
        if not mode.upper() in _allowed_mode:
            raise TypeError('unknown mode: ' + mode)
        self.mode = mode.upper()
        self.size = to_list(size)
        self.pitch, self.channel_pitch = self._get_frame_pitch()
        super(RawFile, self).__init__(path, rewind)

    def _get_frame_pitch(self):
        """Get bytes length of one frame.
        For the detail of mode fourcc, please see https://www.fourcc.org/

        RGB, BGR, and UV channel of NV12, NV21 is packed, while YV12 and YV21 is planar, hence we have:
          - **channel0** of YV12, YV21, NV12, NV21 if Y
          - **channel1** of YV12 is U, of YV21 is V, of NV12 is UV, of NV21 is VU
          - **channel2** of YV12 is V, of YV21 is U
        """
        mode = self.mode
        width, height = self.size
        if mode in ('YV12', 'YV21'):
            return height * width * 3 // 2, [height * width, height * width // 4, height * width // 4]
        if mode in ('NV12', 'NV21'):
            return height * width * 3 // 2, [height * width, height * width // 2]
        if mode in ('RGB', 'BGR'):
            return height * width * 3, [height * width * 3]

    def _get_frame_channel_shape(self):
        """Get each channel's shape according to mode and frame length.
        For the detail of mode fourcc, please see https://www.fourcc.org/
        """
        mode = self.mode
        width, height = self.size
        if mode in ('YV12', 'YV21'):
            return np.array([1, height, width]), np.array([1, height // 2, width // 2]), np.array(
                [1, height // 2, width // 2])
        if mode in ('NV12', 'NV21'):
            return np.array([1, height, width]), np.array([2, height // 2, width // 2])
        if mode in ('RGB', 'BGR'):
            return np.array([height, width, 3])

    def read_frame(self, frames=1, *args):
        """
        read number `frames` of the file.

        Arguments:
            frames: number of frames to be loaded
            id: specify frame format to store (default gray-scale)

        Raise:

        """
        if self.mode in ('YV12', 'YV21', 'NV12', 'NV21',):
            # discard uv plain for acceleration
            _image_mode = 'L'
        else:
            _image_mode = 'RGB'
        return [Image.frombytes(_image_mode, self.size, self.read(self.pitch)) for _ in range(frames)]

    @property
    def shape(self):
        return self.size

    @property
    def frames(self):
        return (self.end_pointer - self.read_pointer) // self.pitch


class ImageFile(File):

    def __init__(self, path, rewind):
        """Open image file or a sequence of image files

        Args:
            path: file path or handle
            rewind: rewind the file when reaches EOF
        """
        super(ImageFile, self).__init__(path, rewind)

    def read_frame(self, frames=1, *args):
        """read number `frames` of the file.

        Args:
            frames: number of frames to be loaded
        """
        image_bytes = [BytesIO(self.read()) for _ in range(frames)]
        return [Image.open(fp) for fp in image_bytes]

    @property
    def shape(self):
        with Image.open(self.file[0]) as img:
            return img.shape

    @property
    def frames(self):
        return len(self.file)
