"""
Copyright: Intel.Corp 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Jan. 11th 2018
Updated Date: May 24th 2018

offline dataset collector
support random crop
"""
import json
from pathlib import Path
from ..Util.Utility import to_list


class Dataset:
    """Dataset provides training/validation/testing data for neural network.

    This is a simple wrapper provides train, val, test and additional properties
    """

    def __init__(self, **kwargs):
        self._args = kwargs
        # default attr
        self._args['mode'] = 'pil-image' if not 'mode' in kwargs else kwargs['mode']
        self._args['depth'] = 1 if not 'depth' in kwargs else kwargs['depth']

    def __getattr__(self, item):
        if item in self._args:
            return self._args[item]
        else:
            if item in ('train', 'val', 'test'):
                raise ValueError(f'un{item}able')
            return None

    def __setitem__(self, key, value):
        self._args[key] = value

    def setattr(self, **kwargs):
        for key in kwargs:
            self._args[key] = kwargs[key]


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
    url_r = url.relative_to(url_p)
    if str(url_r) == '.':
        return url_p.iterdir()
    return url_p.glob(str(url_r))


def load_datasets(json_file):
    """load dataset described in JSON file"""

    datasets = {}
    with open(json_file, 'r') as fd:
        config = json.load(fd)
        all_set_path = config["Path"]
        for name, value in config["Dataset"].items():
            assert isinstance(value, dict)
            datasets[name] = Dataset()
            for i in ('train', 'val', 'test'):
                sets = []
                for j in to_list(value[i]):
                    try:
                        sets += list(_glob_absolute_pattern(all_set_path[j]))
                    except KeyError:
                        sets += list(_glob_absolute_pattern(j))
                datasets[name].__setitem__(i, sets)
            if 'param' in value:
                for k, v in value['param'].items():
                    datasets[name].__setitem__(k, v)
    return datasets


# @deprecated
class Dataset1(object):
    """globs all images in the data path"""

    def __init__(self, train, val, test, train_filter, val_filter, test_filter, **kwargs):
        """Specify data path

        Args:
           :param train: training data
           :param val: validation data
           :param test: testing data
           :param train_filter, val_filter, test_filter: filename filters
        """
        self._train = self._preprocess(train, train_filter)
        self._val = self._preprocess(val, val_filter)
        self._test = self._preprocess(test, test_filter)
        self._args = kwargs

    @staticmethod
    def _preprocess(url, filters):
        if not url or not Path(url).exists():
            return
        if not isinstance(filters, list):
            raise TypeError("Filters is not in list!")
        files = []
        for _f in filters:
            files += [str(x) for x in Path(url).glob(_f)]
        return files

    @property
    def args(self, *args):
        return [self._args[i] for i in args]

    def __getattr__(self, item):
        if item == 'train_size':
            return len(self._train)
        elif item == 'val_size':
            return len(self._val)
        elif item == 'test_size':
            return len(self._test)
        elif item == 'train':
            if not self._train:
                raise ValueError('Untrainable')
            else:
                return self._train
        elif item == 'val':
            if not self._val:
                raise ValueError('Unvalidatable')
            else:
                return self._val
        elif item == 'test':
            if not self._test:
                raise ValueError('Untestable')
            else:
                return self._test
        elif item in self._args:
            return self._args[item]
        else:
            return None


# @deprecated
def load_datasets1(json_file='datasets.json'):
    DATASET = dict()
    with open(json_file, 'r') as fd:
        config = json.load(fd)
        for k, v in config.items():
            _train, _train_filter = None, []
            _val, _val_filter = None, []
            _test, _test_filter = None, []
            if 'train' in v:
                _train = v['train']['url']
                _train_filter = v['train']['filter'].split(',')
            if 'val' in v:
                _val = v['val']['url']
                _val_filter = v['val']['filter'].split(',')
            if 'test' in v:
                _test = v['test']['url']
                _test_filter = v['test']['filter'].split(',')
            for _k in ('train', 'val', 'test'):
                if _k in v.keys():
                    v.pop(_k)
            DATASET[k] = Dataset(
                _train, _val, _test,
                train_filter=_train_filter,
                val_filter=_val_filter,
                test_filter=_test_filter,
                **v)
    return DATASET
