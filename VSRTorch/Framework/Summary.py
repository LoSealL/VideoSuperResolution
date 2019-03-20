#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019 - 3 - 20

from tensorboardX import SummaryWriter

_writer_container = {}


class Summarizer:
  def __init__(self, log_path, key=None):
    if key is not None:
      self.key = hash(key)
    else:
      self.key = hash(str(log_path))
    self._logd = log_path
    self.writer = SummaryWriter(str(log_path))
    _writer_container[self.key] = self

  def close(self):
    self.writer.close()

  def scalar(self, name, x, step=None, collection=None):
    if collection is not None:
      name = f'{collection}/{name}'
    self.writer.add_scalar(name, x, step)

  def image(self, name, image, max=3, step=None, collection=None):
    if image.ndimension() == 4:
      images = image.split(image.shape[0])[:max]
    else:
      assert image.ndimension() == 3
      images = [image]
    if collection is not None:
      name = f'{collection}/{name}'
    for i, img in enumerate(images):
      self.writer.add_image(f'{name}_{i}', img, step)

  def tensor(self, name, tensor, max=3, step=None, reshape=None):
    assert tensor.ndimension() == 4, "Support 4-D tensor only!"
    shape = tensor.shape

    def _placement(t):
      if t <= 16:
        return 4, t // 4
      elif t <= 64:
        return 8, t // 8
      elif t <= 256:
        return 16, t // 16
      else:
        return 32, t // 32

    if reshape:
      col, row = reshape
    else:
      col, row = _placement(shape[1])
    tensor = tensor.view([shape[0], row, col, shape[2], shape[3]])
    tensor = tensor.transpose(2, 3)
    tensor = tensor.view([shape[0], row * shape[2], 1, col * shape[3], 1])
    tensor = tensor.squeeze([2, 4])
    tensor = tensor.unsqueeze(1)
    self.image(name, tensor, step, max, collection='features')


def get_writer(key) -> Summarizer:
  return _writer_container.get(hash(key))
