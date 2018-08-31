"""
Copyright: Intel Corp. 2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: May 9th 2018
Updated Date: May 9th 2018

virtual file is an abstraction of a file or
a collection of ordered frames
"""
from pathlib import Path
from io import SEEK_SET, SEEK_CUR, SEEK_END, BytesIO
from PIL import Image
import numpy as np

from ..Util.Utility import to_list
from ..Framework.Motion import open_flo, KITTI


class File:
    """An abstract file object

    NOTE: If `path` is a file, `File` opens it and calculates its length.
    If `path` is a folder, `File` opens every file in the folder in alphabet order.

    Args:
         path: path to a node (can be a file or a folder contains multiple files).
         rewind: rewind the file automatically when reaches EOF.
    """

    def __init__(self, path, rewind=False):
        self.path = Path(path)
        self.file = []
        self.length = dict()
        self.name = self.path.stem
        if self.path.is_file():
            self.file = [self.path]
            self.length[self.path.name] = self.path.stat().st_size
        elif self.path.is_dir():
            for _file in self.path.glob('*'):
                self.file.append(_file)
                self.length[_file.name] = _file.stat().st_size
        self.read_file = []
        self.read_pointer = 0
        self.end_pointer = sum(self.length.values())
        self.cur_fd = None
        self.rewind = rewind

    def __len__(self):
        """total size of file or files"""
        return self.end_pointer

    def _seek(self, target):
        """seek to a target position of `File`

        Args:
            target: an int representing byte position.
        """
        assert 0 <= target < self.end_pointer
        if self.read_pointer == target:
            # do not need operation
            return
        if not self.cur_fd and self.file:
            # guarantee an opened fd, happens if `File` is just initialized.
            self.cur_fd = self.file[0].open('rb')
            self.read_file.append(self.file.pop(0))
        while self.read_pointer < target:
            # move forward
            reminder = self.length.get(self.read_file[-1].name) - self.cur_fd.tell()
            if self.read_pointer + reminder >= target:
                self.cur_fd.seek(target - self.read_pointer, SEEK_CUR)
                self.read_pointer = target
                return
            else:
                self.read_pointer += reminder
                self.cur_fd.close()
                self.cur_fd = self.file[0].open('rb')
                self.read_file.append(self.file.pop(0))
        while self.read_pointer > target:
            # move backward
            reminder = self.cur_fd.tell()
            if self.read_pointer - reminder <= target:
                self.cur_fd.seek(target - self.read_pointer, SEEK_CUR)
                self.read_pointer = target
                return
            else:
                self.read_pointer -= reminder
                self.cur_fd.close()
                self.cur_fd = self.read_file[-1].open('rb')
                self.file.insert(0, self.read_file.pop())

    def reopen(self):
        """clear the current state and re-initialize read pointer"""
        self.file = self.read_file + self.file
        self.read_file.clear()
        self.read_pointer = 0
        self.cur_fd = None

    def split(self, depth):
        raise NotImplementedError

    def read(self, count=None):
        """Read `count` bytes

        Args:
            count: size of bytes to read, if None (default), read all bytes of current file

        Return:
            bytes: bytes read
        """
        if count == 0:
            return b''
        if not self.cur_fd and self.file:
            self.cur_fd = self.file[0].open('rb')
            self.read_file.append(self.file.pop(0))
        elif not self.cur_fd:
            raise FileNotFoundError('No frames in File')
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
                self.read_file.append(self.file.pop(0))
                return self.read(count)
            elif self.rewind and self.read_file:
                self.reopen()
                return self.read(count)
            else:
                raise EOFError('End of File!')

    def read_frame(self, frames=1, *args):
        """An abstract interface"""
        raise NotImplementedError

    def seek(self, offset, where):
        """Seek the position by `offset` relative to `where`.

        Args:
             offset: move the read pointer by `offset` bytes.
             where: same as io.SEEK_END, io.SEEK_CUR or io.SEEK_SET.
        """
        if where == SEEK_SET:
            self._seek(offset)
        if where == SEEK_CUR:
            self._seek(self.read_pointer + offset)
        if where == SEEK_END:
            self._seek(self.end_pointer + offset)

    def tell(self):
        """Tell the current position of the read pointer."""
        return self.read_pointer

    def size(self, name):
        """Get the length of the file named `name`

        Args:
            name: specify a named file.

        Return:
            int: length in bytes
        """
        path = Path(name)
        name = path.stem if path.exists() else name
        return self.length.get(name)


# Supported RAW format, see 'FOURCC' standard
_ALLOWED_RAW_FORMAT = [
    'YV12',  # [Y][U][V]
    'YV21',  # [Y][V][U]
    'NV12',  # [Y][UV]
    'NV21',  # [Y][VU]
    'RGB',  # [RGB]
    'BGR',  # [BGR]
    'RGBA',  # [RGBA]
    'BGRA',  # [BGRA]
]


class RawFile(File):
    """For reading raw files. The file is lazy loaded, which means
    the file is opened but not loaded into memory at initialization.

    Args:
         path: a string representing `node` path.
         mode: a string, since raw file has no headers, type must be
           explicitly given, see `_ALLOWED_RAW_FORMAT`.
         size: a tuple of int (width, height). If `path` is a folder,
           all files in it must be the same shape.
         rewind: rewind the file automatically when reaches EOF

    Raise:
        TypeError: if `mode` is not supported
    """

    def __init__(self, path, mode, size, rewind=False):

        if not mode.upper() in _ALLOWED_RAW_FORMAT:
            raise TypeError('unknown mode: ' + mode)
        self.mode = mode.upper()
        self.size = to_list(size)
        self.pitch, self.channel_pitch = self._get_frame_pitch()
        super(RawFile, self).__init__(path, rewind)

    def _get_frame_pitch(self):
        """Get bytes length of one frame.
        For the detail of mode fourcc, please see https://www.fourcc.org/

        NOTE: RGB, BGR, and UV channel of NV12, NV21 is packed, while YV12 and YV21 is planar, hence we have:
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
        if mode in ('RGBA', 'BGRA'):
            return height * width * 4, [height * width * 4]

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
        if mode in ('RGBA', 'BGRA'):
            return np.array([height, width, 4])

    def read_frame(self, frames=1, *args):
        """read number of `frames` of the file. A frame is a single image

        Args:
            frames: number of frames to be loaded.
        """
        if self.mode in ('YV12', 'YV21', 'NV12', 'NV21',):
            # TODO discard uv plain for acceleration
            _image_mode = 'L'
            return [Image.frombytes(_image_mode, self.size, self.read(self.pitch)) for _ in range(frames)]
        elif self.mode in ('RGB', 'RGBA'):
            _image_mode = self.mode
            return [Image.frombytes(_image_mode, self.size, self.read(self.pitch)) for _ in range(frames)]
        elif self.mode in ('BGR',):
            _image_mode = 'RGB'
            return [Image.frombytes(
                _image_mode, self.size, b''.join((self.read(3)[::-1] for _ in range(self.pitch // 3))))
                for _ in range(frames)]
        elif self.mode in ('BGRA',):
            _image_mode = 'RGBA'
            img = []
            for _ in range(frames):
                buf = bytes()
                for _ in range(self.pitch // 4):
                    c = self.read(4)
                    buf.join((c[2::-1], c[3:]))
                img.append(Image.frombytes(_image_mode, self.size, buf))
            return img

    def seek(self, offset, where=SEEK_SET):
        """Seek the position by `offset` relative to `where`.

        Args:
             offset: move the read pointer by `offset` bytes.
             where: same as io.SEEK_END, io.SEEK_CUR or io.SEEK_SET.
        """
        if where == SEEK_SET:
            super(RawFile, self).seek(offset * self.pitch, where)
        if where == SEEK_CUR:
            super(RawFile, self).seek(offset * self.pitch - self.tell() % self.pitch, where)
        if where == SEEK_END:
            super(RawFile, self).seek(offset * self.pitch, where)

    @property
    def shape(self):
        return self.size

    @property
    def frames(self):
        """unread frames remain in `RawFile`"""
        return (self.end_pointer - self.read_pointer) // self.pitch


class ImageFile(File):
    """Open image file or a sequence of image frames

    Args:
        path: a string representing `node` path.
        rewind: rewind the file when reaches EOF.
    """

    def __init__(self, path, rewind=False):
        super(ImageFile, self).__init__(path, rewind)
        self._flow = None

    def read_frame(self, frames=1, *args):
        """read number `frames` of the file.

        Args:
            frames: number of frames to be loaded
        """
        image_bytes = [BytesIO(self.read()) for _ in range(frames)]
        return [Image.open(fp) for fp in image_bytes]

    def seek(self, offset, where=SEEK_SET):
        """Seek the position by `offset` relative to `where`.

        Args:
            offset: move the read pointer by `offset` bytes.
            where: same as io.SEEK_END, io.SEEK_CUR or io.SEEK_SET.
        """
        if where == SEEK_CUR:
            cur = len(self.read_file)
            pos = cur + offset
        elif where == SEEK_END:
            pos = len(self.read_file) + len(self.file) + offset
        else:
            pos = offset
        self.reopen()
        self.read_file = self.file[:pos]
        self.file = self.file[pos:]
        self.cur_fd = None

    def attach_flow(self, flow_file):
        self._flow = flow_file
        return self

    @property
    def flow(self):
        fd = Path(self._flow)
        assert fd.exists()
        if fd.suffix == '.flo':
            flow = open_flo(str(fd))
        elif fd.suffix == '.png':
            flow = KITTI.open_flow(str(fd))
        else:
            raise TypeError('unsupported flow format', fd.suffix)
        return flow

    @property
    def shape(self):
        with Image.open(self.file[0]) as img:
            return img.width, img.height

    @property
    def frames(self):
        return len(self.file)
