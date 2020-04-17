#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import re
from concurrent import futures
from pathlib import Path
import copy

import yaml

from .VirtualFile import ImageFile, RawFile
from ..Util import Config, to_list

try:
  from yaml import FullLoader as _Loader
except ImportError:
  from yaml import Loader as _Loader

IMAGE_SUF = ('PNG', 'JPG', 'JPEG', 'BMP', 'TIFF', 'TIF', 'GIF')
VIDEO_SUF = {
  'NV12': 'NV12',
  'YUV': 'YV12',
  'YV12': 'YV12',
  'NV21': 'NV21',
  'YV21': 'YV21',
  'RGB': 'RGB'
}


def _supported_image(x: Path):
  return x.suffix[1:].upper() in IMAGE_SUF


def _supported_video(x: Path):
  return x.suffix[1:].upper() in VIDEO_SUF


def _supported_suffix(x: Path):
  return _supported_image(x) or _supported_video(x)


class Dataset(object):
  """ Make a `dataset` object

  """

  def __init__(self, *folders):
    self.dirs = list(map(Path, folders))
    self.recursive = True
    self.glob_patterns = ('*',)
    self.inc_patterns = None
    self.exc_patterns = None
    self.as_video = False
    self.compiled = None

  def use_like_video_(self):
    self.as_video = True

  def use_like_video(self):
    d = copy.copy(self)
    d.compiled = None
    d.use_like_video_()
    return d

  def include_(self, *pattern: str):
    self.glob_patterns = list(pattern)
    self.inc_patterns = None

  def include(self, *pattern: str):
    d = copy.copy(self)
    d.compiled = None
    d.include_(*pattern)
    return d

  def include_reg_(self, *reg: str):
    self.inc_patterns = [re.compile(r) for r in reg]
    self.glob_patterns = ('*',)

  def include_reg(self, *reg: str):
    d = copy.copy(self)
    d.compiled = None
    d.include_reg_(*reg)
    return d

  def exclude_(self, *reg: str):
    self.exc_patterns = [re.compile(r) for r in reg]

  def exclude(self, *reg: str):
    d = copy.copy(self)
    d.compiled = None
    d.exclude_(*reg)
    return d

  def compile(self):
    if self.compiled:
      return self.compiled
    files = []

    def _exc(x: Path):
      if self.exc_patterns:
        for reg in self.exc_patterns:
          if reg.search(str(x.absolute().as_posix())):
            return False
      return True

    def _inc(x: Path):
      if self.inc_patterns:
        for reg in self.inc_patterns:
          if reg.search(str(x.absolute().as_posix())):
            return True
      return False

    for folder in self.dirs:
      if not Path(folder).exists():
        continue
      nodes = []
      if folder.is_file():
        # if points to a file rather than a directory
        nodes.append(folder)
      fn_glob = Path.rglob if self.recursive else Path.glob
      for pat in self.glob_patterns:
        nodes += list(fn_glob(folder, pat))
      if self.inc_patterns:
        nodes = filter(_inc, nodes)
      files += list(filter(_exc, filter(_supported_suffix, nodes)))
    image_nodes = list(filter(_supported_image, files))
    if not self.as_video:
      self.compiled = Container(sorted(image_nodes), self.as_video)
      return self.compiled
    video_nodes = list(filter(_supported_video, files))
    video_nodes += list(map(lambda x: x.parent, image_nodes))
    video_nodes = list(set(video_nodes))  # remove duplicated nodes
    self.compiled = Container(sorted(video_nodes), self.as_video)
    return self.compiled


class Container(object):
  """Frames container

  """

  def __init__(self, urls, is_video: bool):
    assert isinstance(urls, (list, tuple))
    pool = futures.ThreadPoolExecutor(4)
    fs = []
    self.nodes = []

    def _parse_image_node(url: Path):
      if url.is_dir():
        for i in filter(_supported_image, url.glob('*')):
          self.nodes.append(ImageFile(i, rewind=True))
      elif _supported_image(url):
        self.nodes.append(ImageFile(url, rewind=True))

    def _parse_video_node(url: Path):
      if _supported_video(url):
        size = re.findall("\\d+x\\d+", url.stem)
        if size:
          size = [int(x) for x in size[0].split('x')]
          self.nodes.append(
              RawFile(url, VIDEO_SUF[url.suffix[1:].upper()], size,
                      rewind=True))
      elif url.is_dir():
        self.nodes.append(ImageFile(url))

    for j in urls:
      if is_video:
        fs.append(pool.submit(_parse_video_node, j))
      else:
        fs.append(pool.submit(_parse_image_node, j))
    futures.as_completed(fs)
    pool.shutdown()
    self.nodes = sorted(self.nodes, key=lambda x: x.path)

  def __getitem__(self, item):
    return self.nodes[item]

  def __len__(self):
    return len(self.nodes)

  @property
  def capacity(self):
    if not self.nodes:
      return 0
    pos = 0
    max_sz = 0
    total_frames = 0
    for i, n in enumerate(self.nodes):
      total_frames += n.frames
      if n.size() > max_sz:
        max_sz = n.size()
        pos = i
    shape = self.nodes[pos].shape
    max_bpp = 3
    return shape[0] * shape[1] * max_bpp * total_frames


def load_datasets(describe_file, key=''):
  """load dataset described in YAML file"""

  def _extend_pattern(url):
    _url = root / Path(url)
    url_p = _url

    while True:
      try:
        if url_p.exists():
          break
      except OSError:
        url_p = url_p.parent
        continue
      if url_p == url_p.parent:
        break
      url_p = url_p.parent
    # retrieve glob pattern
    url_r = str(_url.relative_to(url_p))
    if url_r == '.' and url_p.is_dir():
      return str(Path(url) / '**/*')
    return url

  def _get_dataset(desc, use_as_video=None, name=None):
    dataset = Config(name=name)
    for i in desc:
      if i not in ('train', 'val', 'test'):
        continue
      if isinstance(desc[i], dict):
        hr = to_list(desc[i].get('hr'))
        lr = to_list(desc[i].get('lr'))
      else:
        hr = to_list(desc[i])
        lr = []
      if use_as_video:
        hr_pattern = [
          x if x not in all_path and x + '[video]' not in all_path else
          all_path[x + '[video]'] for x in hr]
        lr_pattern = [
          x if x not in all_path and x + '[video]' not in all_path else
          all_path[x + '[video]'] for x in lr]
      else:
        hr_pattern = [x if x not in all_path else all_path[x] for x in hr]
        lr_pattern = [x if x not in all_path else all_path[x] for x in lr]
      hr_data = Dataset(root).include(*(_extend_pattern(x) for x in hr_pattern))
      lr_data = Dataset(root).include(
          *(_extend_pattern(x) for x in lr_pattern)) if lr_pattern else None
      hr_data.recursive = False
      if lr_data is not None:
        lr_data.recursive = False
      if use_as_video:
        hr_data.use_like_video_()
        if lr_data is not None:
          lr_data.use_like_video_()
      setattr(dataset, i, Config(hr=hr_data, lr=lr_data))
    return dataset

  datasets = Config()
  with open(describe_file, 'r') as fd:
    config = yaml.load(fd, Loader=_Loader)
    root = Path(config["Root"])
    if not root.is_absolute():
      # make `root` relative to the file
      root = Path(describe_file).resolve().parent / root
      root = root.resolve()
    all_path = config["Path"]
    if key.upper() in config["Dataset"]:
      return _get_dataset(config["Dataset"][key.upper()], name=key)
    elif key.upper() + '[video]' in config["Dataset"]:
      return _get_dataset(config["Dataset"][key.upper() + '[video]'], True,
                          name=key)
    elif key.upper() in all_path:
      return _get_dataset(Config(test=all_path[key.upper()]), name=key)
    elif key.upper() + '[video]' in all_path:
      return _get_dataset(Config(test=all_path[key.upper() + '[video]']), True,
                          name=key)
    for name, value in config["Dataset"].items():
      if '[video]' in name:
        name = name.replace('[video]', '')
        datasets[name] = _get_dataset(value, True, name=name)
      else:
        datasets[name] = _get_dataset(value, name=name)
    for name in all_path:
      if '[video]' in name:
        _name = name.replace('[video]', '')
        datasets[_name] = _get_dataset(Config(test=all_path[name]), True,
                                       name=_name)
      else:
        datasets[name] = _get_dataset(Config(test=all_path[name]), name=name)
    return datasets
