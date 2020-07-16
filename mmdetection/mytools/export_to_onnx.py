"""
    @lichenyang 2019.11.17
    Refer to demo/webcam_demo.py

    Generate the predicted bboxes for all test imgs, write the result into the corresponding txt.
"""

import argparse

import os
import os.path as osp
import cv2
import pdb
import torch
import numpy as np
from tqdm import tqdm
import json

from mmdet.apis import inference_detector, init_detector, show_result, export_onnx

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection network inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--image_path', 
                      help='path to load images for export',
                      default="")
    parser.add_argument('--onnx_save_path', 
                      help='directory to load images for demo',  
                      default="model.onnx")
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument('--output_names', type=str, default=None, nargs='+', 
                        help='onnx model output names')

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    model = init_detector(args.config, args.checkpoint, device=torch.device('cuda', args.device))
    export_onnx(model, args.image_path, args.onnx_save_path, output_names=args.output_names)


if __name__ == '__main__':
    main()
