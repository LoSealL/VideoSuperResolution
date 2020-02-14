#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/5/25 下午4:38

import logging

_logger = logging.getLogger("VSR.RBPN")
_logger.info("LICENSE: RBPN is implemented by M. Haris, et. al. @alterzero")
_logger.warning(
  "I use unsupervised flownet to estimate optical flow, rather than pyflow module.")
