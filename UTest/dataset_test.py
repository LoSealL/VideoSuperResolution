from VSR.DataLoader.Dataset import _glob_absolute_pattern, load_datasets

try:
    DATASETS = load_datasets('./Data/datasets.yaml')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.yaml')


def test_glob_absolute_pattern():
    URL = './data/set5_x2'
    node = _glob_absolute_pattern(URL)
    assert len(node) == 5
    assert node[0].match('img_001_SRF_2_LR.png')
    assert node[1].match('img_002_SRF_2_LR.png')
    assert node[2].match('img_003_SRF_2_LR.png')
    assert node[3].match('img_004_SRF_2_LR.png')
    assert node[4].match('img_005_SRF_2_LR.png')

    URL = './data'
    node = _glob_absolute_pattern(URL)
    assert len(node) == 3
    assert node[0].match('flying_chair')
    assert node[1].match('kitti_car')
    assert node[2].match('set5_x2')

    URL = './data/flying_chair/*.flo'
    node = _glob_absolute_pattern(URL)
    assert len(node) == 1
    assert node[0].match('0-gt.flo')

    URL = './data/**/*.png'
    node = _glob_absolute_pattern(URL)
    assert len(node) == 10


def test_existence():
    _K = DATASETS.keys()
    for k in _K:
        print('==== [', k, '] ====')
        _V = []
        try:
            _V = DATASETS[k].train
        except ValueError:
            if not _V:
                print('Train set of', k, 'doesn\'t exist.')
        finally:
            _V = []
        try:
            _V = DATASETS[k].val
        except ValueError:
            if not _V:
                print('Val set of', k, 'doesn\'t exist.')
        finally:
            _V = []
        try:
            _V = DATASETS[k].test
        except ValueError:
            if not _V:
                print('Test set of', k, 'doesn\'t exist.')
        print('=========================', flush=True)


def test_dataset_class():
    pass