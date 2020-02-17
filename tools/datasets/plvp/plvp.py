from random import shuffle
import numpy as np
from datasets.BaseDataset import BaseDataset
import cv2


class PLVP(BaseDataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(PLVP, self).__init__(setting, split_name, preprocess, file_length)
        self.gt_down_sampling = setting['gt_down_sampling']

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        # Select which textfile (contains the filenames) to use
        source = self._train_source
        if split_name == "val":
            source = self._eval_source
        elif split_name == 'test':
            source = self._test_source

        file_names = []
        with open(source, 'r') as f:
            files = f.read().splitlines()

        # Split into portions
        if self._portion is not None:
            shuffle(files)
            num_files = len(files)
            if self._portion > 0:
                split = int(np.floor(self._portion * num_files))
                files = files[:split]
            elif self._portion < 0:
                split = int(np.floor((1 + self._portion) * num_files))
                files = files[split:]

        # To follow the author original design
        file_names = [[item, item] for item in files]

        return file_names

    def _fetch_data(self, img_path, gt_path, dtype=None):

        img = self._open_image(img_path + '.jpg',
                               down_sampling=self._down_sampling)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        try:
            gt = self._open_image(gt_path + '_lane.png', cv2.IMREAD_GRAYSCALE,
                                  dtype=dtype,
                                  down_sampling=self._down_sampling)
        except:
            gt = self._open_image(gt_path + '_lane.PNG', cv2.IMREAD_GRAYSCALE,
                                  dtype=dtype,
                                  down_sampling=self._down_sampling)

        # Model output before final outsampling
        gt = cv2.resize(gt, (
            gt.shape[0] // self.gt_down_sampling,
            gt.shape[1] // self.gt_down_sampling),
                        interpolation=cv2.INTER_NEAREST)

        return img, gt
