#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 14

import numpy as np

def psnr(x, y, max_val=1.0):
  mse = np.square(x - y).mean()
  return 10 * np.log10(max_val**2 / mse)
