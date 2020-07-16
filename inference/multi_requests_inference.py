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
from collections import defaultdict

from openvino.inference_engine import IENetwork, IECore

from utils.infer_request_wrap import InferRequestsQueue
from utils.retinanet_adapter import RetinaNetAdapter
from utils.nms import nms
from utils.tools import Timer


class DataLoader(object):
    def __init__(self, image_dir, strides, patch_size):
        self.image_dir = image_dir
        self.strides = strides
        self.patch_size = patch_size
        self.image_names = os.listdir(image_dir)
        
        self.image_name = ''
        self.image_id = 0
        self.x_idx = 0
        self.y_idx = 0
        self.x_num = 0
        self.y_num = 0
        self.x_idx_y_idx_list = []

    def next(self):
        if len(self.x_idx_y_idx_list) == 0:
            success = self.read_image_and_update_info()
            if not success:
                return None

        self.x_idx, self.y_idx = self.x_idx_y_idx_list.pop(0)

        x = self.strides[0] * self.x_idx if self.x_idx < self.x_num - 1 else self.image_shape[0] - self.patch_size[0]
        y = self.strides[1] * self.y_idx if self.y_idx < self.y_num - 1 else self.image_shape[1] - self.patch_size[1]

        crop_img = self.image[:, 
                              y : y + self.patch_size[1], 
                              x : x + self.patch_size[0]].copy()
        crop_img = crop_img[np.newaxis, :, :, :]
        crop_meta = dict(image_name=self.image_name, x=x, y=y)

        # self.x_idx = min(self.x_idx + 1, self.x_num)
        # self.y_idx = min(self.y_idx + 1, self.y_num)

        # log.info('id: {}, image_name: {}, x: {}, y: {}'.format(self.image_id, self.image_name, x, y))
        return dict(img=crop_img, meta=crop_meta)


    def read_image_and_update_info(self):
        if self.image_id == len(self.image_names):
            return False

        image_name = self.image_names[self.image_id]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)

        height, width, _ = image.shape
        image_shape = (width, height)
        self.x_num, self.y_num = self.calc_split_num(image_shape, self.patch_size, self.strides)
        self.x_idx, self.y_idx = 0, 0 

        self.image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        self.image_name = image_name
        self.image_shape = image_shape
        self.image_id += 1

        self.x_idx_y_idx_list.clear()
        for i in range(self.x_num):
            for j in range(self.y_num):
                self.x_idx_y_idx_list.append((i, j))

        return True

    def calc_split_num(self, image_shape, patch_size, strides):
        #strides = [cfg.patch_size[0]//2, cfg.patch_size[1]//2]
        #x_num = (image_shape[0] - cfg.patch_size[0]) // strides[0] + 2
        #y_num = (image_shape[1] - cfg.patch_size[1]) // strides[1] + 2
        x_num = math.ceil((image_shape[0] - patch_size[0]) / strides[0]) + 1
        y_num = math.ceil((image_shape[1] - patch_size[1]) / strides[1]) + 1
        return x_num, y_num



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
    log.info('input shape: {}'.format(input_info.shape))

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')
    assert (len(net.outputs.keys()) == 2), "Sample supports topologies only with 2 output"

    loc_out_name = "797"
    class_out_name =  "741"
    assert (loc_out_name in net.outputs.keys()) and (class_out_name in net.outputs.keys())

    loc_out_info = net.outputs[loc_out_name]
    class_out_info = net.outputs[class_out_name]

    loc_out_info.precision = "FP32"
    class_out_info.precision = "FP32"
    # -----------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------
    log.info("Loading model to the device")
    # cpu_throughput = {'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
    ie.set_config({'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}, args.device)
    ie.set_config({'CPU_BIND_THREAD': 'YES'}, args.device)
    exec_net = ie.load_network(network=net, device_name=args.device, num_requests=0)

    infer_requests = exec_net.requests
    request_queue = InferRequestsQueue(infer_requests)
    log.info('nreqs: {}, nstream:{}'.format(len(infer_requests), ie.get_config(args.device, 'CPU_THROUGHPUT_STREAMS')))

    # --------------------------- 3. Read and preprocess input --------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    if args.voc_res_file and os.path.exists(args.voc_res_file):
        os.remove(args.voc_res_file)
    

    load_data_timer = Timer(name='load_data')
    post_process_timer = Timer(name='post_process')

    adapter = RetinaNetAdapter(input_shape=args.patch_size)

    # --------------------------- Performing inference ----------------------------------------------------
    result_all_images = defaultdict(list)
    data_loader = DataLoader(args.image_dir, args.strides, args.patch_size)
    while True:
        load_data_timer.tic()
        input_data = data_loader.next()
        load_data_timer.toc()

        if input_data == None:
            break

        infer_request = request_queue.get_idle_request()
        if not infer_request:
            raise Exception('No idle Infer Requests!')

        if infer_request.cur_meta == None:
            infer_request.start_async(input_name, input_data)
            continue
        
        # get result
        post_process_timer.tic()
        image_name = infer_request.cur_meta['image_name']
        x = infer_request.cur_meta['x']
        y = infer_request.cur_meta['y']

        loc_out = infer_request.request.outputs[loc_out_name][0]
        class_out = infer_request.request.outputs[class_out_name][0]

        ## start infer
        infer_request.start_async(input_name, input_data)

        ## post-process
        result = adapter.process(loc_out, class_out)
        result, _ = nms(result, thresh=0.5, keep_top_k=100)

        result[:, 0] += x
        result[:, 1] += y
        result[:, 2] += x
        result[:, 3] += y
        result_all_images[image_name].append(result)
        post_process_timer.toc()


    # wait the latest inference executions
    request_queue.wait_all()
    post_process_timer.tic()
    for infer_request in request_queue.requests:
        # get result
        image_name = infer_request.cur_meta['image_name']
        x = infer_request.cur_meta['x']
        y = infer_request.cur_meta['y']

        loc_out = infer_request.request.outputs[loc_out_name][0]
        class_out = infer_request.request.outputs[class_out_name][0]

        ## post-process
        result = adapter.process(loc_out, class_out)
        result, _ = nms(result, thresh=0.5, keep_top_k=100)

        result[:, 0] += x
        result[:, 1] += y
        result[:, 2] += x
        result[:, 3] += y
        result_all_images[image_name].append(result)
    post_process_timer.toc()


    post_process_timer.tic()
    ## process total image result
    for image_name, result_per_image in result_all_images.items():
        result_per_image = np.concatenate(result_per_image, axis=0)
        nms_result, _ = nms(result_per_image, thresh=0.5)

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

            if args.voc_res_file:
                xmin = x
                ymin = y
                xmax = int(nms_result[i, 2])
                ymax = int(nms_result[i, 3])
                voc_str = voc_format.format(os.path.splitext(image_name)[0], p, xmin, ymin, xmax, ymax)
                voc_all.append(voc_str)

        file_name = os.path.splitext(image_name)[0] + '.json'
        with open(os.path.join(args.result_dir, file_name), 'w') as f:
            json.dump(pos_all, f)

        if args.voc_res_file:
            with open(args.voc_res_file, 'a') as f:
                for voc_str in voc_all:
                    f.write(voc_str+'\n')
    
    post_process_timer.toc()
    total_timer.toc()
    # -----------------------------------------------------------------------------------------------------
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
    all_timers.extend([load_data_timer,
                       post_process_timer,
                       total_timer])
    for timer in all_timers:
        log.info('{}: avg: {:.2f} ms, total: {:.2f}s'.format(timer.name, timer.avg * 1000, timer.total))

    log.info('infer: {:2f}s'.format(request_queue.get_duration_in_seconds()))
    log.info("Execution successful\n")


if __name__ == '__main__':
    sys.exit(main() or 0)



