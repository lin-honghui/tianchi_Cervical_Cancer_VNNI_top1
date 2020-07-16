import os
import mmcv
import cv2
from mmdet.datasets import CervicalCancerDataset
from mmdet.datasets.pipelines import Compose

# dataset_type = 'CervicalCancerDataset'
data_root = 'data/Cervical_Cancer/'
ann_file=data_root + 'train/label'
img_prefix=data_root + 'train/image'

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_norm_cfg = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CervicalCancerCrop', crop_size=(1600, 1600)),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    #dict(type='RandomShiftGtBBox', shift_rate=0.2),
    dict(type='ReplaceBackground', 
        drop_rate=1., 
        crop_size=(800, 800),
        background_dir='data/Cervical_Cancer/neg_roi'
        ),
    #dict(type='RandomFlip', flip_ratio=0),
    #dict(type='RandomVerticalFlip', flip_ratio=0),
    #dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    # dict(type='DefaultFormatBundle'),
    #dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


trans_pipeline = [
   dict(type='ReplaceBackground', 
       drop_rate=1., 
       crop_size=(800, 800),
       background_dir='data/Cervical_Cancer/neg_roi'
       ),
   #dict(type='RandomShiftGtBBox', shift_rate=1.)
]

cervical_cancer_dataset = CervicalCancerDataset(ann_file, 
                                                train_pipeline, 
                                                img_prefix=img_prefix,
                                                test_mode=False)

# idx = 0                                                
# data = cervical_cancer_dataset.prepare_train_img(idx)
trans_pipeline = Compose(trans_pipeline)

save_dir = 'tests/vis_imgs'
img_infos = cervical_cancer_dataset.img_infos

for idx in range(len(img_infos)):
    #if img_infos[idx]['filename'] == 'T2019_837_roi_0.png':
    #    data = cervical_cancer_dataset.prepare_train_img(idx)
   

    data = cervical_cancer_dataset.prepare_train_img(idx)
    if data == None:
        continue
    #ori_img = data['img']
    #gt_bboxes = data['gt_bboxes'].astype('int')
    #
    #trans_data = trans_pipeline(data)
    #trans_img = trans_data['img']
    #trans_gt_bboxes = trans_data['gt_bboxes'].astype('int')

    #for bbox, trans_gt_bbox in zip(gt_bboxes, trans_gt_bboxes):
    #    xmin, ymin, xmax, ymax = bbox
    #    cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
    #    
    #    cv2.rectangle(trans_img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
    #    xmin, ymin, xmax, ymax = trans_gt_bbox 
    #    cv2.rectangle(trans_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)

    ##import pdb;pdb.set_trace()
    #image_name = os.path.split(data['filename'])[-1]
    #ori_save_path = os.path.join(save_dir, 'ori_' + image_name)
    #trans_save_path = os.path.join(save_dir, 'trans_' + image_name)
    #cv2.imwrite(ori_save_path, ori_img)
    #cv2.imwrite(trans_save_path, trans_img)

    #print(idx)
    #print(ori_save_path)
    if idx == 5:
        break


