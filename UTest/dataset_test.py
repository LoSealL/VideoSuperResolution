from VSR.DataLoader.Dataset import _glob_absolute_pattern, load_datasets

try:
    DATASETS = load_datasets('./Data/datasets.json')
except FileNotFoundError:
    DATASETS = load_datasets('../Data/datasets.json')


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
