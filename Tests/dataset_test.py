import os

if not os.getcwd().endswith('Tests'):
  os.chdir('Tests')
from VSR.DataLoader.Dataset import Dataset, load_datasets


def test_image_data():
  d = Dataset('data/set5_x2')
  data = d.compile()
  assert len(data) is 5
  assert data.capacity == 983040


def test_video_data():
  d = Dataset('data/video/custom_pair').use_like_video()
  data = d.compile()
  assert len(data) is 2


def test_multi_url():
  d = Dataset('data/set5_x2', 'data/kitti_car')
  data = d.compile()
  assert len(data) is 8


def test_include_exclude():
  d = Dataset('data')
  d.include_('xiuxian*')
  data1 = d.compile()
  d = Dataset('data')
  d.include_reg_('set5')
  data2 = d.compile()
  d = Dataset('data').include_reg('set5').exclude('png')
  data3 = d.compile()

  assert len(data1) is 6
  assert len(data2) is 5
  assert len(data3) is 0


def test_dataset_desc_file():
  ddf = 'data/fake_datasets.yml'
  datasets = load_datasets(ddf)
  assert len(datasets) is 9
  assert len(datasets.NONE.train.hr.compile()) is 0
  assert len(datasets.NORMAL.train.hr.compile()) is 7
  assert len(datasets.NORMAL.val.hr.compile()) is 5
  assert len(datasets.NORMAL.test.hr.compile()) is 1
  assert len(datasets.PAIR.train.hr.compile()) is 2
  assert len(datasets.PAIR.train.lr.compile()) is 2
  assert len(datasets.VIDEOPAIR.train.hr.compile()) is 1
  assert len(datasets.VIDEOPAIR.train.lr.compile()) is 1
  assert len(datasets.VIDEOPAIR.val.hr.compile()) is 1
  assert len(datasets.VIDEOPAIR.val.lr.compile()) is 1
  assert len(datasets.VIDEOPAIR.test.hr.compile()) is 1
  assert len(datasets.VIDEOPAIR.test.lr.compile()) is 1
  assert len(datasets.FOO.test.hr.compile()) is 2
  assert len(datasets.BAR.test.hr.compile()) is 5
  assert datasets.VIDEOPAIR.train.hr.as_video
  assert datasets.XIUXIAN.test.hr.as_video

  raw = load_datasets(ddf, 'RAW')
  assert len(raw.train.hr.compile()) is 1
  assert len(raw.val.hr.compile()) is 1
  assert len(raw.test.hr.compile()) is 1
  assert raw.train.hr.as_video


if __name__ == '__main__':
  test_image_data()
  test_video_data()
  test_multi_url()
  test_include_exclude()
  test_dataset_desc_file()
