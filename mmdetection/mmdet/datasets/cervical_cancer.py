"""
  @lichenyang 2019.11.16
  The cervical cancer dataset.
"""

import os
import cv2
import json
import pickle
import numpy as np
import os.path as osp

from .custom import CustomDataset
from .registry import DATASETS

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


@DATASETS.register_module
class CervicalCancerDataset(CustomDataset):
    CLASSES = ('pos', )

    def load_annotations(self, ann_file):
        # cache_path = "./data/cache"
        # cache_file = osp.join(cache_path, self.__class__.__name__ + '_gt_roidb.pkl')

        # # Cache file already exist.
        # if osp.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         img_infos = pickle.load(fid)
        #     print('{} gt roidb loaded from {}'.format(self.__class__.__name__, cache_file))
        #     return img_infos

        json_names = os.listdir(ann_file)
        img_infos = [self._load_annotation(ann_file, json_name) for json_name in json_names]

        # if not osp.exists(cache_path):
        #     os.makedirs(cache_path)
        # with open(cache_file, 'wb') as fid:
        #     pickle.dump(img_infos, fid, pickle.HIGHEST_PROTOCOL)
        # print('wrote gt roidb to {}'.format(cache_file))

        return img_infos



    def _load_annotation(self, ann_file, json_name):
        json_path = osp.join(ann_file, json_name)
        with open(json_path, 'r') as f:
            ann = json.load(f)
            
            roi_x  = ann['x']
            roi_y  = ann['y']
            width  = ann['w']
            height = ann['h']

            pos_list = ann['pos_list']
            pos_nums = len(pos_list)

        gt_bboxes = np.zeros((pos_nums, 4), dtype=np.float32)
        gt_labels = np.zeros((pos_nums), dtype=np.int64)
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        for idx, pos in enumerate(pos_list):
            x1 = max(pos['x']-roi_x, 0)
            y1 = max(pos['y']-roi_y, 0)
            x2 = min(x1+pos['w']-1, width)
            y2 = min(y1+pos['h']-1, height)

            cls = self._class_to_ind(pos['class'])
            gt_bboxes[idx, :] = [x1, y1, x2, y2]
            gt_labels[idx] = cls

        file_name = json_name.replace('.json', '.png')
        ann = dict(
            filename=file_name,
            height=height,
            width=width,
            ann=dict(
                bboxes=gt_bboxes,
                labels=gt_labels,
                bboxes_ignore=gt_bboxes_ignore,
                )
            )

        return ann

    
    def _class_to_ind(self, class_name):
        return CervicalCancerDataset.CLASSES.index(class_name) + 1