#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import math
import json
import numpy as np
import logging as log
from time import time
from collections import defaultdict

from openvino.inference_engine import IENetwork, IECore
from utils.retinanet_adapter import RetinaNetAdapter
from utils.nms import nms
from utils.tools import Timer

from queue import Queue
import random,threading,time

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
        required=True, type=str)
    args.add_argument("-i", "--image_dir", help="Required. Path to image file.",
        required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
        help="Optional. Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.",
        type=str, default=None)
    args.add_argument("-d", "--device",
        help="Optional. Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)",
        default="CPU", type=str)

    args.add_argument("--loc_out_name", help="location output name",
                      required=True, type=str)
    args.add_argument("--class_out_name", help="classification output name",
                      required=True, type=str)
    args.add_argument("-r", "--result_dir", help="Required. Path to result file.",
                      required=True, type=str)
    args.add_argument('--patch_size',
                      nargs='+', type=int, help='patch_size')
    args.add_argument('--strides',
                      nargs='+', type=int, help='strides')
    parser.add_argument('--voc_res_file',
                      help='file to save result in voc format',
                      default=None)

    return parser


def calc_split_num(image_shape, patch_size, strides):
    #strides = [cfg.patch_size[0]//2, cfg.patch_size[1]//2]
    #x_num = (image_shape[0] - cfg.patch_size[0]) // strides[0] + 2
    #y_num = (image_shape[1] - cfg.patch_size[1]) // strides[1] + 2
    x_num = math.ceil((image_shape[0] - patch_size[0]) / strides[0]) + 1
    y_num = math.ceil((image_shape[1] - patch_size[1]) / strides[1]) + 1
    return x_num, y_num



#生产者类
class Producer(threading.Thread):
    def __init__(self, name, queue, args):
        threading.Thread.__init__(self, name=name)
        self.data = queue
        self.args = args

        self.strides = args.strides
        self.patch_size = args.patch_size
        self.image_dir = args.image_dir

    def run(self):
        producer_timer = Timer(name='producer')
        producer_timer.tic()

        image_names = os.listdir(self.image_dir)
        log.info("image_nums: {}".format(len(image_names)))
        for image_id, image_name in enumerate(image_names):
            image_path = os.path.join(self.image_dir, image_name)
            img = cv2.imread(image_path).astype('float32')

            height, width, _ = img.shape
            image_shape = (width, height)

            x_num, y_num = calc_split_num(image_shape, self.patch_size, self.strides)

            log.info("id:{}, name: {}, shape: ({},{}), x_num:{}, y_num:{}".format(
                    image_id, image_name, height, width, x_num, y_num))

            # n = 1
            img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW

            for i in range(x_num):
                for j in range(y_num):
                    x = self.strides[0] * i if i < x_num - 1 else image_shape[0] - self.patch_size[0]
                    y = self.strides[1] * j if j < y_num - 1 else image_shape[1] - self.patch_size[1]
                    # print('[Producer]processing {} , x: {}, y: {}'.format(image_name, x, y))

                    crop_img = img[:, y:(y+self.patch_size[1]), x:(x+self.patch_size[0])].copy()
                    crop_img = crop_img[np.newaxis, :, :, :]
                    crop_img_meta=dict(name=image_name, x=x, y=y)
                    self.data.put(dict(img=crop_img, meta=crop_img_meta))

        self.data.put('quit')
        producer_timer.toc()
        log.info('{}: avg: {:.2f} ms, total: {:.2f}s'.format(producer_timer.name, producer_timer.avg * 1000, producer_timer.total))


#消费者类
class Consumer(threading.Thread):
    def __init__(self,name,queue, exec_net, input_name, args):
        threading.Thread.__init__(self,name=name)
        self.data=queue
        self.exec_net = exec_net
        self.input_name = input_name
        self.loc_out_name = args.loc_out_name
        self.class_out_name = args.class_out_name
        self.result_dir = args.result_dir
        self.voc_res_file = args.voc_res_file

        self.result_all_images = defaultdict(list)
        self.adapter = RetinaNetAdapter(input_shape=args.patch_size)

    def run(self):
        # --------------------------- Performing inference ----------------------------------------------------
        infer_timer = Timer(name='infer')
        adapter_timer = Timer(name='adapter')
        patch_img_nms_timer = Timer(name='patch_img_nms')
        add_offset_timer = Timer(name='add_offset')
        whole_img_nms_timer = Timer(name='whole_img_nms')
        while True:
            
            data = self.data.get()
            if data=='quit':
                break
            else:
                crop_img, crop_img_meta = data['img'], data['meta']
                image_name = crop_img_meta['name']
                x = crop_img_meta['x']
                y = crop_img_meta['y']
            # import pdb;pdb.set_trace()
            print('[Producer]forward {} , x: {}, y: {}'.format(crop_img_meta['name'], 
                                                               crop_img_meta['x'], 
                                                               crop_img_meta['y']))

            infer_timer.tic()
            res = self.exec_net.infer(inputs={self.input_name: crop_img})
            loc_out = res[self.loc_out_name][0]
            class_out = res[self.class_out_name][0]
            infer_timer.toc()

            adapter_timer.tic()
            result = self.adapter.process(loc_out, class_out)
            adapter_timer.toc()

            patch_img_nms_timer.tic()
            result, _ = nms(result, thresh=0.5, keep_top_k=100)
            patch_img_nms_timer.toc()

            # # import pdb;pdb.set_trace()
            add_offset_timer.tic()
            result[:, 0] += x
            result[:, 1] += y
            result[:, 2] += x
            result[:, 3] += y
            self.result_all_images[image_name].append(result)
            add_offset_timer.toc()

        for image_name, result_per_image in self.result_all_images.items():
            whole_img_nms_timer.tic()
            result_per_image = np.concatenate(result_per_image, axis=0)
            nms_result, _ = nms(result_per_image, thresh=0.5)
            whole_img_nms_timer.toc()

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

                if self.voc_res_file:
                    xmin = x
                    ymin = y
                    xmax = int(nms_result[i, 2])
                    ymax = int(nms_result[i, 3])
                    voc_str = voc_format.format(os.path.splitext(image_name)[0], p, xmin, ymin, xmax, ymax)
                    voc_all.append(voc_str)

            file_name = os.path.splitext(image_name)[0] + '.json'
            with open(os.path.join(self.result_dir, file_name), 'w') as f:
                json.dump(pos_all, f)

            if self.voc_res_file:
                with open(self.voc_res_file, 'a') as f:
                    for voc_str in voc_all:
                        f.write(voc_str+'\n')


        all_timers = []
        all_timers.extend([infer_timer, 
                           adapter_timer, 
                           patch_img_nms_timer, 
                           whole_img_nms_timer, 
                           add_offset_timer])
        for timer in all_timers:
            log.info('{}: avg: {:.2f} ms, total: {:.2f}s'.format(timer.name, timer.avg * 1000, timer.total))

def main():
    total_timer = Timer(name='total')
    total_timer.tic()

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    # --------------------------- 1. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    # -----------------------------------------------------------------------------------------------------

    # ------------- 2. Load Plugin for inference engine and extensions library if specified --------------
    log.info("Loading Inference Engine")
    ie = IECore()
    log.info("Device info:")
    versions = ie.get_versions(args.device)
    print("{}{}".format(" "*8, args.device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[args.device].major, versions[args.device].minor))
    print("{}Build ........... {}".format(" "*8, versions[args.device].build_number))
    
    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
        log.info("CPU extension loaded: {}".format(args.cpu_extension))

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    # -----------------------------------------------------------------------------------------------------


    # --------------------------- 4. Configure input & output ---------------------------------------------
    # --------------------------- Prepare input blobs -----------------------------------------------------
    log.info("Preparing input blobs")
    assert (len(net.inputs.keys()) == 1), "Sample supports topologies only with 1 input"

    input_name = next(iter(net.inputs.keys()))
    input_info = net.inputs[input_name]
    input_info.precision = 'FP32'

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')
    assert (len(net.outputs.keys()) == 2), "Sample supports topologies only with 2 output"

    loc_out_name = args.loc_out_name
    class_out_name =  args.class_out_name
    assert (loc_out_name in net.outputs.keys()) and (class_out_name in net.outputs.keys())

    loc_out_info = net.outputs[loc_out_name]
    class_out_info = net.outputs[class_out_name]

    loc_out_info.precision = "FP32"
    class_out_info.precision = "FP32"
    # -----------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------
    log.info("Loading model to the device")
    exec_net = ie.load_network(network=net, device_name=args.device)


    # # --------------------------- 3. Read and preprocess input --------------------------------------------
    # # -----------------------------------------------------------------------------------------------------
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    if args.voc_res_file and os.path.exists(args.voc_res_file):
        os.remove(args.voc_res_file)
    
    # create_anchor_timer = Timer(name='create_anchor')
    # read_img_timer = Timer(name='read_img')
    # preprocess_timer = Timer(name='preprocess')
    # infer_timer = Timer(name='infer')
    # adapter_timer = Timer(name='adapter')
    # patch_img_nms_timer = Timer(name='patch_img_nms')
    # whole_img_nms_timer = Timer(name='whole_img_nms')
    # add_offset_timer = Timer(name='add_offset')
    # write_result_timer = Timer(name='write_result')

    queue = Queue()
    producer = Producer('Producer', queue, args)
    consumer = Consumer('Consumer', queue, exec_net, input_name, args)

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()
    log.info('All threads finished!')

        
    total_timer.toc()
    # # -----------------------------------------------------------------------------------------------------
    all_timers = []
    # all_timers.extend([create_anchor_timer,
    #                    read_img_timer,
    #                    preprocess_timer, 
    #                    infer_timer, 
    #                    adapter_timer, 
    #                    patch_img_nms_timer, 
    #                    whole_img_nms_timer, 
    #                    add_offset_timer,
    #                    write_result_timer, 
    #                    total_timer])
    all_timers.extend([total_timer])
    for timer in all_timers:
        log.info('{}: avg: {:.2f} ms, total: {:.2f}s'.format(timer.name, timer.avg * 1000, timer.total))

    log.info("Execution successful\n")


if __name__ == '__main__':
    sys.exit(main() or 0)