#  Copyright (c) 2017-2020 Wenyi Tang.
#  Author: Wenyi Tang
#  Email: wenyitang@outlook.com
#  Update: 2020 - 2 - 10

from VSR.DataLoader import Dataset, Loader, load_datasets
from VSR.Util import Config, save_inference_images
from common import parse_arguments, parser

g1 = parser.add_argument_group("evaluating options")
g1.add_argument("-t", "--test", nargs='*', help="specify test dataset name or data path")
g1.add_argument("--ensemble", action="store_true")
g1.add_argument("--video", action="store_true", help="notify load test data as video stream")
g2 = parser.add_argument_group("advanced options")
g2.add_argument("--output_index", default='-1', help="specify access index of output array (slicable)")
g2.add_argument("--export", help="export ONNX (torch backend) or protobuf (tf backend) (needs support from model)")
g2.add_argument("--overwrite", action="store_true", help="overwrite the existing predicted output files")
g2.add_argument('--depth', type=int, default=1, help="frame sequence")


def main():
  args = parse_arguments()
  opt = args.opt
  root = args.root
  model = args.model

  datasets = load_datasets(args.data_config_file)
  try:
    test_datas = [datasets[t.upper()] for t in opt.test] if opt.test else []
  except KeyError:
    test_datas = [Config(test=Config(lr=Dataset(*opt.test)), name='infer')]
    if opt.video:
      test_datas[0].test.lr.use_like_video_()
  # enter model executor environment
  with model.get_executor(root) as t:
    for data in test_datas:
      run_benchmark = False if data.test.hr is None else True
      if run_benchmark:
        ld = Loader(data.test.hr, data.test.lr, opt.scale,
                    threads=opt.threads)
      else:
        ld = Loader(data.test.hr, data.test.lr, threads=opt.threads)
      if opt.channel == 1:
        # convert data color space to grayscale
        ld.set_color_space('hr', 'L')
        ld.set_color_space('lr', 'L')
      config = t.query_config(opt)
      config.inference_results_hooks = [save_inference_images(root / data.name, opt.output_index, not opt.overwrite)]
      config.batch_shape = [1, opt.depth, -1, -1, -1]
      config.traced_val = True
      if run_benchmark:
        t.benchmark(ld, config)
      else:
        t.infer(ld, config)
    if opt.export:
      t.export(opt.export)


if __name__ == '__main__':
  main()
