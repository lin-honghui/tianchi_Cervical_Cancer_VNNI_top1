import argparse

import os
import cv2
import math
import pdb
import torch
import numpy as np
from tqdm import tqdm
import json
import time

import mmcv
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.apis import inference_detector, init_detector, show_result
from mmdet.ops.nms.nms_wrapper import nms


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection network inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',  
                      default="images")
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument('--result_dir',
                      help='directory to write result txt for demo',
                      default="pred")
    parser.add_argument('--voc_res_file',
                      help='file to save result in voc format',
                      default="voc_res_file.txt")
    parser.add_argument('--patch_size',
                      nargs='+', type=int, help='patch_size')
    parser.add_argument('--strides',
                      nargs='+', type=int, help='strides')
    parser.add_argument('--score_thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
    parser.add_argument('--vis_image_dir', dest='vis_image_dir',
                      help='directory to write vis images for demo',
                      default="/home/LiChenyang/download/Cervical_Cancer_20191114/visualization/image20191114_test_res101-fpn_vis")
    parser.add_argument('--ann_dir',
                      help='directory to load annotation for visualize',
                      default="")

    args = parser.parse_args()
    return args


def load_annotation(ann_file):
    with open(ann_file, 'r') as f:
        ann = json.load(f)
        
        roi_x  = ann['x']
        roi_y  = ann['y']
        width  = ann['w']
        height = ann['h']

        pos_list = ann['pos_list']
        pos_nums = len(pos_list)

    gt_bboxes = np.zeros((pos_nums, 4), dtype=np.float32)

    for idx, pos in enumerate(pos_list):
        x1 = max(pos['x']-roi_x, 0)
        y1 = max(pos['y']-roi_y, 0)
        x2 = min(x1+pos['w'], width)
        y2 = min(y1+pos['h'], height)

        gt_bboxes[idx, :] = [x1, y1, x2, y2]

    return gt_bboxes


def calc_split_num(image_shape, patch_size, strides):
    #strides = [cfg.patch_size[0]//2, cfg.patch_size[1]//2]
    #x_num = (image_shape[0] - cfg.patch_size[0]) // strides[0] + 2
    #y_num = (image_shape[1] - cfg.patch_size[1]) // strides[1] + 2
    x_num = math.ceil((image_shape[0] - patch_size[0]) / strides[0]) + 1
    y_num = math.ceil((image_shape[1] - patch_size[1]) / strides[1]) + 1
    return x_num, y_num


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device=torch.device('cuda', args.device))
    cfg = model.cfg
    # # build the data pipeline
    test_pipeline = cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    device = next(model.parameters()).device  # model device

    image_names = os.listdir(args.image_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if os.path.exists(args.voc_res_file):
        os.remove(args.voc_res_file)
    if args.vis:
        if not os.path.exists(args.vis_image_dir):
            os.makedirs(args.vis_image_dir)

    print("Begin to predict mask: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for image_name in image_names:
        image_path = os.path.join(args.image_dir, image_name)
        img = mmcv.imread(image_path)

        height, width, _ = img.shape
        image_shape = (width, height)
        strides = args.strides
        patch_size = args.patch_size
        x_num, y_num = calc_split_num(image_shape, patch_size, strides)

        # print(image_name)
        result_all = []
        for i in range(x_num):
            for j in range(y_num):
                x = strides[0] * i if i < x_num - 1 else image_shape[0] - args.patch_size[0]
                y = strides[1] * j if j < y_num - 1 else image_shape[1] - args.patch_size[1]

                crop_img = img[y:y+patch_size[1], x:x+patch_size[0], :].copy()
                data = dict(filename=image_name, 
                            img=crop_img, 
                            img_shape=crop_img.shape, 
                            ori_shape=img.shape)
                data = test_pipeline(data)
                data = scatter(collate([data], samples_per_gpu=1), [device])[0]
                # forward the model
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data)[0]

                result[:, 0] += x
                result[:, 1] += y
                result[:, 2] += x
                result[:, 3] += y
                result_all.append(result)

        # import pdb;pdb.set_trace()
        result_all = np.concatenate(result_all, axis=0)
        nms_result, _ = nms(result_all, 0.5, device_id=args.device)
        # nms_result = result_all

        if args.vis:
            out_file = os.path.join(args.vis_image_dir, image_name)
            vis_img = show_result(img,
                                [nms_result],
                                model.CLASSES,
                                score_thr=args.score_thr,
                                wait_time=0,
                                show=False,
                                out_file=None)

            ann_file = os.path.join(args.ann_dir, image_name.replace('png', 'json'))
            gt_bboxes =load_annotation(ann_file)
            for gt_bbox in gt_bboxes:
                xmin, ymin, xmax, ymax = gt_bbox
                cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            cv2.imwrite(out_file, vis_img)

        voc_format = '{} {:.4f} {} {} {} {}'
        pos_all = []
        voc_all = []
        for i in range(nms_result.shape[0]):
            x = int(nms_result[i, 0])
            y = int(nms_result[i, 1])
            w = max(int(nms_result[i, 2] - nms_result[i, 0]), 1)
            h = max(int(nms_result[i, 3] - nms_result[i, 1]), 1)
            p = float(nms_result[i, 4])
            pos = {'x': x, 'y': y, 'w': w, 'h': h, 'p': p}
            pos_all.append(pos)

            xmin = x
            ymin = y
            xmax = int(nms_result[i, 2])
            ymax = int(nms_result[i, 3])
            voc_str = voc_format.format(os.path.splitext(image_name)[0], p, xmin, ymin, xmax, ymax)
            voc_all.append(voc_str)

        with open(os.path.join(args.result_dir, image_name.replace('png', 'json')), 'w') as f:
            json.dump(pos_all, f)

        with open(args.voc_res_file, 'a') as f:
            for voc_str in voc_all:
                f.write(voc_str+'\n')

        print("Finish predict mask: ", image_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    main()
