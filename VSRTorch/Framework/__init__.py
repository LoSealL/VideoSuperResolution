#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 14

import logging

_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
hdl = logging.StreamHandler()
hdl.setLevel(logging.DEBUG)
hdl.setFormatter(_FORMATTER)

std_logger = logging.getLogger("VSR")
std_logger.setLevel(logging.DEBUG)
std_logger.addHandler(hdl)
