#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

import easydict
import yaml

try:
  from yaml import FullLoader as _Loader
except ImportError:
  from yaml import Loader as _Loader


class Config(easydict.EasyDict):
  def __init__(self, obj=None, **kwargs):
    super(Config, self).__init__(**kwargs)
    if obj is not None:
      assert isinstance(obj, (dict, str))
      if isinstance(obj, str):
        with open(obj, 'r') as fd:
          obj = yaml.load(fd, Loader=_Loader)
      self.update(**obj)

  def __getattr__(self, item):
    return self.get(item)
