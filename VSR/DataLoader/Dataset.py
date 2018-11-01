"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Jan. 11th 2018
Updated Date: May 24th 2018

offline dataset collector
support random crop
"""
from pathlib import Path
from ..Util.Utility import to_list
from ..Util.Config import Config
import yaml


class Dataset(Config):
    """Dataset provides training/validation/testing data for neural network.

    This is a simple wrapper provides train, val, test and additional properties

    Args:
        train: a list of file path, representing train set.
        val: a list of file path, representing validation set.
        test: a list of file path, representing test set.
        infer: a list of file path, representing infer set.
        mode: a string representing data format. 'pil-image' for formatted
          image, or ('YV12', 'NV12', 'RGB', 'BGR'...) for raw data.
          See `VirtualFile._ALLOWED_RAW_FORMAT`
        flow: a list of file path representing optical flow files
    """

    def __init__(self, **kwargs):
        super(Dataset, self).__init__(kwargs)
        # default attr
        if self.mode is None:
            self.mode = 'pil-image1'

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            if item in ('train', 'val', 'test', 'infer',):
                import tensorflow as tf
                tf.logging.debug(f'The {item} files is empty!')
                return []
            return None


def _glob_absolute_pattern(url):
    url = Path(url)
    url_p = url
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
    url_r = url.relative_to(url_p)
    if url_p.is_dir():
        if str(url_r) == '.':
            # url is a folder contains only folders
            ret = url_p.iterdir()
        else:
            # glob pattern
            ret = url_p.glob(str(url_r))
    else:
        # iff url is a single file
        ret = [url_p]
    # sort is necessary for `glob` behaves differently on UNIX/Windows
    ret = to_list(ret)
    ret.sort()
    return ret


def load_datasets(describe_file):
    """load dataset described in YAML file"""

    datasets = {}
    with open(describe_file, 'r') as fd:
        config = yaml.load(fd)
        root = Path(config["Root"])
        all_set_path = config["Path"]
        all_set_path.update(config["Path_Tracked"])
        for name, value in config["Dataset"].items():
            assert isinstance(value, dict)
            datasets[name] = Dataset(name=name)
            for i in value:
                if i not in ('train', 'val', 'test', 'infer', 'flow'):
                    continue
                sets = []
                for j in to_list(value[i]):
                    try:
                        sets += _glob_absolute_pattern(root / all_set_path[j])
                    except KeyError:
                        sets += _glob_absolute_pattern(j)
                setattr(datasets[name], i, sets)
            if 'param' in value:
                for k, v in value['param'].items():
                    setattr(datasets[name], k, v)
        for name, path in config["Path_Tracked"].items():
            if name not in datasets:
                datasets[name] = Dataset(name=name)
                datasets[name].test = _glob_absolute_pattern(root / path)
    return datasets
