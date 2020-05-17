#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 7

from functools import partial
from pathlib import Path

from .ImageProcess import array_to_img


def _str_to_slice(index: str):
  index = index.split(':')
  if len(index) == 1:
    ind = int(index[0])
    if ind < 0:
      sl = slice(ind, None, None)
    else:
      sl = slice(ind, ind + 1)
  else:
    def _maybe_int(x):
      try:
        return int(x)
      except ValueError:
        return None

    sl = slice(*(_maybe_int(i) for i in index))
  return sl


def _save_model_predicted_images(output, names, save_dir, index, auto_rename):
  assert len(names) == 1, f"Name list exceeds 1, which is {names}"
  name = names[0]
  for img in output[_str_to_slice(index)]:
    shape = img.shape
    path = Path(save_dir)
    if shape[0] > 1 or auto_rename:
      path /= name
    path.mkdir(exist_ok=True, parents=True)
    for i, n in enumerate(img):
      rep = 0
      if auto_rename:
        while (path / f"{name}_id{i:04d}_{rep:04d}.png").exists():
          rep += 1
      path /= f"{name}_id{i:04d}_{rep:04d}.png"
      array_to_img(n).convert('RGB').save(str(path))
  return output


def save_inference_images(save_dir, multi_output_index=-1, auto_rename=None):
  return partial(_save_model_predicted_images, save_dir=save_dir,
                 index=multi_output_index, auto_rename=auto_rename)
