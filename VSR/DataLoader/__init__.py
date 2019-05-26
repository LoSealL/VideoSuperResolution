#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import logging
import os

__all__ = [
  'Dataset',
  'Loader',
  'VirtualFile'
]

__options__ = {
  # Set to True to enable lazy load. May speed up the first loading time if your
  # dataset contains much plenty of files.
  'VSR_LAZY_LOAD': '',
  # TODO Test: Saving memory
  'VSR_CUSTOM_PAIR_AGGR_MEM': '',
}

_logger = logging.getLogger('VSR.Loader')
for i in __options__:
  # preset environment or default value
  os.environ[i] = os.getenv(i) or __options__[i]
