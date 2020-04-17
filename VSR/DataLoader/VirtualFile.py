#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

from io import BytesIO, SEEK_CUR, SEEK_END, SEEK_SET
from pathlib import Path

import numpy as np
from PIL import Image

from . import NVDecoder, YVDecoder
from .FloDecoder import open_flo, KITTI

Image.register_decoder('NV12', NVDecoder.NV12Decoder)
Image.register_decoder('NV21', NVDecoder.NV21Decoder)
Image.register_decoder('YV12', YVDecoder.YV12Decoder)
Image.register_decoder('YV21', YVDecoder.YV21Decoder)


class File:
  """An abstract file object

  NOTE: If `path` is a file, `File` opens it and calculates its length.
  If `path` is a folder, `File` opens every file in the folder in abc order.

  Args:
       path: path to a **node**, where node can be a **file** or a **folder**
         contains multiple files.
       rewind: rewind the file automatically when reaches EOF.
  """

  def __init__(self, path, rewind=False):
    self.path = Path(path)
    self.file = []
    self.length = dict()
    self.name = self.path.stem
    self.full_name = self.path.absolute().as_posix()
    if self.path.is_file():
      self.file = [self.path]
      self.length[self.path.name] = self.path.stat().st_size
    elif self.path.is_dir():
      for _file in self.path.glob('*'):
        self.file.append(_file)
        self.length[_file.name] = _file.stat().st_size
      # sort the files by name, because they are unordered in UNIX
      self.file.sort()
    self.file_ = self.file.copy()
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
      reminder = self.length.get(self.read_file[-1].name) - \
                 self.cur_fd.tell()
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
    self.file = self.file_.copy()
    self.read_file.clear()
    self.read_pointer = 0
    self.cur_fd = None

  def split(self, depth):
    raise NotImplementedError

  def read(self, count=None):
    """Read `count` bytes

    Args:
        count: size of bytes to read, if None (default),
          read all bytes of current file

    Return:
        bytes: bytes read
    """
    if count == 0:
      return b''
    if not self.cur_fd and self.file:
      self.cur_fd = self.file[0].open('rb')
      self.read_file.append(self.file.pop(0))
    elif not self.cur_fd:
      if self.rewind and self.read_file:
        self.reopen()
        return self.read(count)
      else:
        raise EOFError(f'End of File! {self.name}')
    read_bytes = self.cur_fd.read(count)
    if read_bytes:
      self.read_pointer += len(read_bytes)
      if count and count > len(read_bytes):
        return read_bytes + self.read(count - len(read_bytes))
      elif count:
        return read_bytes
      else:
        # read entire file, close the file descriptor
        self.cur_fd.close()
        self.cur_fd = None
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
        raise EOFError(f'End of File! {self.name}')

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

  def size(self, name=None):
    """Get the length of the file named `name`

    Args:
        name: specify a named file.

    Return:
        int: length in bytes
    """
    if name is None:
      return sum(self.length.values())
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
    self._size = size
    self.pitch, self.channel_pitch = self._get_frame_pitch()
    super(RawFile, self).__init__(path, rewind)
    self._pair = None

  def _get_frame_pitch(self):
    """Get bytes length of one frame.
    For the detail of mode fourcc, please see https://www.fourcc.org/

    NOTE: RGB, BGR, and UV channel of NV12, NV21 is packed, while YV12 and
      YV21 is planar, hence we have:
      - **channel0** of YV12, YV21, NV12, NV21 if Y
      - **channel1** of YV12 is U, YV21 is V, NV12 is UV, NV21 is VU
      - **channel2** of YV12 is V, YV21 is U
    """
    mode = self.mode
    w, h = self._size
    if mode in ('YV12', 'YV21'):
      return h * w * 3 // 2, [h * w, h * w // 4, h * w // 4]
    if mode in ('NV12', 'NV21'):
      return h * w * 3 // 2, [h * w, h * w // 2]
    if mode in ('RGB', 'BGR'):
      return h * w * 3, [h * w * 3]
    if mode in ('RGBA', 'BGRA'):
      return h * w * 4, [h * w * 4]

  def _get_frame_channel_shape(self):
    """Get each channel's shape according to mode and frame length.
    For the detail of mode fourcc, please see https://www.fourcc.org/
    """
    mode = self.mode
    w, h = self._size
    if mode in ('YV12', 'YV21'):
      return (np.array([1, h, w]),
              np.array([1, h // 2, w // 2]),
              np.array([1, h // 2, w // 2]))
    if mode in ('NV12', 'NV21'):
      return np.array([1, h, w]), np.array([2, h // 2, w // 2])
    if mode in ('RGB', 'BGR'):
      return np.array([h, w, 3])
    if mode in ('RGBA', 'BGRA'):
      return np.array([h, w, 4])

  def read_frame(self, frames=1, *args):
    """read number of `frames` of the file. A frame is a single image

    Args:
        frames: number of frames to be loaded.
    """
    if self.mode in ('YV12', 'YV21', 'NV12', 'NV21',):
      ret = []
      for _ in range(frames):
        data = self.read(self.pitch)
        ret.append(Image.frombytes('YCbCr', self._size, data, self.mode))
      return ret
    elif self.mode in ('RGB', 'RGBA'):
      ret = []
      for _ in range(frames):
        data = self.read(self.pitch)
        ret.append(Image.frombytes(self.mode, self._size, data))
      return ret
    elif self.mode in ('BGR',):
      _image_mode = 'RGB'
      ret = []
      for _ in range(frames):
        data = b''.join(
            (self.read(3)[::-1] for _ in range(self.pitch // 3)))
        ret.append(Image.frombytes('RGB', self._size, data))
      return ret
    elif self.mode in ('BGRA',):
      ret = []
      for _ in range(frames):
        buf = bytes()
        for _ in range(self.pitch // 4):
          c = self.read(4)
          buf.join((c[2::-1], c[3:]))
        ret.append(Image.frombytes('RGBA', self._size, buf))
      return ret

  def seek(self, offset, where=SEEK_SET):
    """Seek the position by `offset` relative to `where`.

    Args:
         offset: move the read pointer by `offset` bytes.
         where: same as io.SEEK_END, io.SEEK_CUR or io.SEEK_SET.
    """
    if where == SEEK_SET:
      super(RawFile, self).seek(offset * self.pitch, where)
    if where == SEEK_CUR:
      super(RawFile, self).seek(
          offset * self.pitch - self.tell() % self.pitch, where)
    if where == SEEK_END:
      super(RawFile, self).seek(offset * self.pitch, where)

  def pad(self, padding):
    """RawFile doesn't support pad for now"""

    print(" [!] warning: pad is not supported in RawFile")

  def attach_pair(self, pair_file):
    self._pair = RawFile(pair_file, self.mode, self._size, self.rewind)
    return self

  @property
  def pair(self):
    return self._pair

  @property
  def shape(self):
    return self._size

  @property
  def frames(self):
    """frames in `RawFile`"""
    return self.end_pointer // self.pitch


class ImageFile(File):
  """Open image file or a sequence of image frames

  Args:
      path: a string representing `node` path.
      rewind: rewind the file when reaches EOF.
  """

  def __init__(self, path, rewind=False):
    super(ImageFile, self).__init__(path, rewind)
    self._flow = None
    self._pair = None

  def read_frame(self, frames=1, *args):
    """read number `frames` of the file.

    Args:
        frames: number of frames to be loaded
    """
    image_bytes = [BytesIO(self.read()) for _ in range(frames)]
    return [Image.open(fp) for fp in image_bytes]

  def read_frame2(self, frames=1, *args):
    """new API, saving memory while loading frames. But will consume a lot of
    file descriptors.

    Args:
        frames: number of frames to be loaded
    """
    imgs = []
    if frames == 0:
      return imgs
    while True:
      if len(self.file) > 0:
        cur_fd = self.file.pop(0)
        imgs.append(Image.open(cur_fd))
        self.read_file.append(cur_fd)
        with open(cur_fd, 'rb') as fd:
          fd.seek(0, SEEK_END)
          self.read_pointer += fd.tell()
      elif self.rewind:
        self.reopen()
      else:
        raise EOFError('End of File!')
      if len(imgs) == frames:
        break
    return imgs

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
    if pos < 0:
      pos = 0
    self.file = self.read_file + self.file
    self.read_file = self.file[:pos]
    self.file = self.file[pos:]
    self.cur_fd = None

  def pad(self, padding):
    """Pad file(s) list in the head and tail.
      Padded file is temperate and will be dropped after `reopen()`

    Args:
      padding: a integer or a list of 2 integers.
    """

    if not isinstance(padding, (list, tuple)):
      padding = [padding, padding]
    else:
      assert len(padding) is 2
    if self.read_file:
      raise RuntimeError(
          "pad must be called when reading cursor is at the beginning.")
    for _ in range(padding[0]):
      self.file.insert(0, self.file[0])
    for _ in range(padding[1]):
      self.file.append(self.file[-1])

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

  def attach_pair(self, pair_file):
    self._pair = ImageFile(pair_file, self.rewind)
    return self

  @property
  def pair(self):
    return self._pair

  @property
  def shape(self):
    if self.file:
      file = self.file[0]
    else:
      file = self.read_file[0]
    with Image.open(file) as img:
      return img.width, img.height

  @property
  def frames(self):
    return len(self.file) + len(self.read_file)
