from VSR.DataLoader.Loader import BatchLoader
from VSR.DataLoader.Dataset import load_datasets
from Models.Srcnn import SRCNN

import numpy as np
import time


if __name__ == '__main__':
    """ Test training 1 epoch """

    model = SRCNN(scale=3)
    model.compile()
    datasets = load_datasets('../Data/datasets.json')
    loader = BatchLoader(50, datasets['91-IMAGE'], 'train', scale=3)
    step = 0
    start = time.time()
    for img_hr, img_lr in loader:
        img_hr = np.squeeze(img_hr, 1)
        img_lr = np.squeeze(img_lr, 1)
        model.train_batch(img_lr, img_hr)
        step += 1
        print('-', end='', flush=True)
        if step % 100 == 0:
            consumed = time.time() - start
            start = time.time()
            print(
                f' Time: {consumed:.4f}s, time per batch: {consumed * 10:.2f}ms/b', flush=True)
    print('Train 1 epoch done!', flush=True)
    # model.export_model_pb()
