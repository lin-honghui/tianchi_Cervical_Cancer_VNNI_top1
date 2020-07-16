import inspect
import os

import albumentations
import mmcv
import cv2
import os.path as osp
import numpy as np
from albumentations import Compose
from imagecorruptions import corrupt
from numpy import random
from glob import glob

import numpy as np
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES

import spams
import json
import random
from . import stain_utils as ut

@PIPELINES.register_module
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = masks

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        else:
            raise ValueError(
                'Invalid flipping direction "{}"'.format(direction))
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [
                    mmcv.imflip(mask, direction=results['flip_direction'])
                    for mask in results[key]
                ]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class RandomVerticalFlip(object):
    """Flip the image & bbox & mask vertically.

    If the input dict contains the key "vertical_flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes vertically.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        h = img_shape[0]
        flipped = bboxes.copy()
        flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
        flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        return flipped

    def __call__(self, results):
        if 'vertical_flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['vertical_flip'] = flip
        if results['vertical_flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'], direction='vertical')
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[::-1, :] for mask in results[key]]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = mmcv.impad(results['img'], self.size)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            padded_masks = [
                mmcv.impad(mask, pad_shape, pad_val=self.pad_val)
                for mask in results[key]
            ]
            results[key] = np.stack(padded_masks, axis=0)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


@PIPELINES.register_module
class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

        # filter out the gt bboxes that are completely cropped
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                gt_bboxes[:, 3] > gt_bboxes[:, 1])
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results['gt_bboxes'] = gt_bboxes[valid_inds, :]
            if 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid_inds]

            # filter and crop the masks
            if 'gt_masks' in results:
                valid_gt_masks = []
                for i in np.where(valid_inds)[0]:
                    gt_mask = results['gt_masks'][i][crop_y1:crop_y2,
                                                     crop_x1:crop_x2]
                    valid_gt_masks.append(gt_mask)
                results['gt_masks'] = valid_gt_masks

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)


@PIPELINES.register_module
class SegResizeFlipPadRescale(object):
    """A sequential transforms to semantic segmentation maps.

    The same pipeline as input images is applied to the semantic segmentation
    map, and finally rescale it by some scale factor. The transforms include:
    1. resize
    2. flip
    3. pad
    4. rescale (so that the final size can be different from the image size)

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        if results['keep_ratio']:
            gt_seg = mmcv.imrescale(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        else:
            gt_seg = mmcv.imresize(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        if results['flip']:
            gt_seg = mmcv.imflip(gt_seg)
        if gt_seg.shape != results['pad_shape']:
            gt_seg = mmcv.impad(gt_seg, results['pad_shape'][:2])
        if self.scale_factor != 1:
            gt_seg = mmcv.imrescale(
                gt_seg, self.scale_factor, interpolation='nearest')
        results['gt_semantic_seg'] = gt_seg
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor={})'.format(
            self.scale_factor)


@PIPELINES.register_module
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        img = results['img']
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={}, '
                     'saturation_range={}, hue_delta={})').format(
                         self.brightness_delta, self.contrast_range,
                         self.saturation_range, self.hue_delta)
        return repr_str


@PIPELINES.register_module
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None,
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def __call__(self, results):
        if random.uniform(0, 1) > self.prob:
            return results

        img, boxes = [results[k] for k in ('img', 'gt_bboxes')]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        boxes = boxes + np.tile((left, top), 2).astype(boxes.dtype)

        results['img'] = expand_img
        results['gt_bboxes'] = boxes

        if 'gt_masks' in results:
            expand_gt_masks = []
            for mask in results['gt_masks']:
                expand_mask = np.full((int(h * ratio), int(w * ratio)),
                                      0).astype(mask.dtype)
                expand_mask[top:top + h, left:left + w] = mask
                expand_gt_masks.append(expand_mask)
            results['gt_masks'] = expand_gt_masks

        # not tested
        if 'gt_semantic_seg' in results:
            assert self.seg_ignore_label is not None
            gt_seg = results['gt_semantic_seg']
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label).astype(gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results['gt_semantic_seg'] = expand_gt_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, to_rgb={}, ratio_range={}, ' \
                    'seg_ignore_label={})'.format(
                        self.mean, self.to_rgb, self.ratio_range,
                        self.seg_ignore_label)
        return repr_str


@PIPELINES.register_module
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, results):
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) *
                        (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels

                if 'gt_masks' in results:
                    valid_masks = [
                        results['gt_masks'][i] for i in range(len(mask))
                        if mask[i]
                    ]
                    results['gt_masks'] = [
                        gt_mask[patch[1]:patch[3], patch[0]:patch[2]]
                        for gt_mask in valid_masks
                    ]

                # not tested
                if 'gt_semantic_seg' in results:
                    results['gt_semantic_seg'] = results['gt_semantic_seg'][
                        patch[1]:patch[3], patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str


@PIPELINES.register_module
class Corrupt(object):

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(corruption={}, severity={})'.format(
            self.corruption, self.severity)
        return repr_str


@PIPELINES.register_module
class Albu(object):

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        """
        Adds custom transformations from Albumentations lib.
        Please, visit `https://albumentations.readthedocs.io`
        to get more information.

        transforms (list): list of albu transformations
        bbox_params (dict): bbox_params for albumentation `Compose`
        keymap (dict): contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): whether to skip the image
                                      if no ann left after aug
        """

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                'type must be a str or valid type, but got {}'.format(
                    type(obj_type)))

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)

            # filter label_fields
            if self.filter_lost_elements:

                results['idx_mapper'] = np.arange(len(results['bboxes']))

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = [
                        results['masks'][i] for i in results['idx_mapper']
                    ]

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transformations={})'.format(self.transformations)
        return repr_str



@PIPELINES.register_module
class AlbuMine(object):

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        """
        Adds custom transformations from Albumentations lib.
        Please, visit `https://albumentations.readthedocs.io`
        to get more information.

        transforms (list): list of albu transformations
        bbox_params (dict): bbox_params for albumentation `Compose`
        keymap (dict): contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): whether to skip the image
                                      if no ann left after aug
        """
        """###
            While the gt label is empty, do aug again until the gt label is not empty.
        """

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                'type must be a str or valid type, but got {}'.format(
                    type(obj_type)))

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        while True:
            # print("Before: ", results['gt_labels'].shape)
            results_cur = self.aug(**results)
            # print("After: ", np.array(results_cur['gt_labels']).shape)
            if len(results_cur['gt_labels']) > 0:
                results = results_cur
                break


        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)

            # filter label_fields
            if self.filter_lost_elements:

                results['idx_mapper'] = np.arange(len(results['bboxes']))

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = [
                        results['masks'][i] for i in results['idx_mapper']
                    ]

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transformations={})'.format(self.transformations)
        return repr_str



@PIPELINES.register_module
class CervicalCancerCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_labels']

        #ori_img = img.copy()
        #save_dir = 'tests/vis_imgs'
        #image_name = os.path.split(results['filename'])[-1]
        #ori_save_path = os.path.join(save_dir, 'ori_' + image_name)
        #patch_save_path = os.path.join(save_dir, 'patch_' + image_name)
        #crop_save_path = os.path.join(save_dir, 'crop_' + image_name)
        #for gt_bbox in results['gt_bboxes']:
        #    xmin, ymin, xmax, ymax = map(int, gt_bbox)
        #    cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0,0,255), 12)
        #cv2.imwrite(ori_save_path, ori_img)

        img, gt_bboxes, gt_labels = self.crop_patch(img, gt_bboxes, gt_labels)
        #img, gt_bboxes, gt_labels, nx, ny, px, py = self.crop_patch(img, gt_bboxes, gt_labels)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = gt_labels

        #cv2.rectangle(ori_img, (nx, ny), (nx+px, ny+py), (255,0,0), 12)
        #cv2.imwrite(patch_save_path, ori_img)

        #crop_img = img.copy()
        #for gt_bbox in results['gt_bboxes']:
        #    xmin, ymin, xmax, ymax = map(int, gt_bbox)
        #    cv2.rectangle(crop_img, (xmin, ymin), (xmax, ymax), (0,0,255), 8)
        #cv2.imwrite(crop_save_path, crop_img)

        return results

    def crop_patch(self, img, bboxes, labels):
        H, W, C = img.shape
        px, py = self.crop_size

        if H < py and W < px:
            pad_img = np.zeros((py, px, 3), dtype=img.dtype)
            pad_img[:H, :W, :] = img
            img = pad_img
        elif H < py:
            pad_img = np.zeros((py, W, 3), dtype=img.dtype)
            pad_img[:H, :, :] = img
            img = pad_img
        elif W < px:            
            pad_img = np.zeros((H, px, 3), dtype=img.dtype)
            pad_img[:, :W, :] = img
            img = pad_img

        H, W, C = img.shape
        obj_num = bboxes.shape[0]
        index = np.random.randint(0, obj_num)

        ux, uy, vx, vy = bboxes[index, :]

        nx = np.random.randint(max(vx - px, 0), min(ux + 1, W - px + 1))
        ny = np.random.randint(max(vy - py, 0), min(uy + 1, H - py + 1))

        patch_coord = np.zeros((1, 4), dtype="int")
        patch_coord[0, 0] = nx
        patch_coord[0, 1] = ny
        patch_coord[0, 2] = nx + px
        patch_coord[0, 3] = ny + py

        index = self.compute_overlap(patch_coord, bboxes, 0.5)
        index = np.squeeze(index, axis=0)

        patch = img[ny: ny + py, nx: nx + px, :]
        bboxes = bboxes[index, :]
        bboxes[:, 0] = np.maximum(bboxes[:, 0] - patch_coord[0, 0], 0)
        bboxes[:, 1] = np.maximum(bboxes[:, 1] - patch_coord[0, 1], 0)
        bboxes[:, 2] = np.minimum(self.crop_size[0] + bboxes[:, 2] - patch_coord[0, 2], self.crop_size[0])
        bboxes[:, 3] = np.minimum(self.crop_size[1] + bboxes[:, 3] - patch_coord[0, 3], self.crop_size[1])

        labels = labels[index]

        return patch, bboxes, labels
        #return patch, bboxes, labels, nx, ny, px, py

    
    def compute_overlap(self, a, b, over_threshold=0.5):
        """
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = area

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        overlap = intersection / ua
        index = overlap > over_threshold
        return index


    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)


@PIPELINES.register_module
class StainNormVahadane(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None
        
    def get_stain_matrix(self, I, threshold=0.8, lamda=0.1):
        """
        Get 2x3 stain matrix. First row H and second row E
        :param I:
        :param threshold:
        :param lamda:
        :return:
        """
        mask = ut.notwhite_mask(I, thresh=threshold).reshape((-1,))
        OD = ut.RGB_to_OD(I).reshape((-1, 3))
        OD = OD[mask]
        if OD.size == 0:
            return None
        dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False, numThreads=1).T

        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]
        dictionary = ut.normalize_rows(dictionary)
        return dictionary

    def fit(self, target):
        # target, _ = ut.standardize_brightness(target)
        self.stain_matrix_target = self.get_stain_matrix(target)

    def target_stains(self):
        return ut.OD_to_RGB(self.stain_matrix_target)

    def transform(self, I):
        standard_I, p = ut.standardize_brightness(I)
        
        stain_matrix_source = self.get_stain_matrix(standard_I)
        source_concentrations = ut.get_concentrations(standard_I, stain_matrix_source)
        
        standard_I = (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(standard_I.shape))).astype(
            np.uint8)

        return np.clip(standard_I * p / 255.0, 0, 255).astype(np.uint8)

    def hematoxylin(self, I):
        I = ut.standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = self.get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H


    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module
class ReplaceBackground(object):
    def __init__(self, drop_rate, crop_size, background_dir):

        self.background_list = glob(osp.join(background_dir, "*.png"))
        self.drop_rate = drop_rate
        self.crop_size = crop_size

    def __call__(self, results):

        if np.random.uniform(0,1) < self.drop_rate:

            normalizer = StainNormVahadane()
            while True:
                bg_img_path = random.sample(self.background_list, 1)[0]
                bg_img = cv2.imread(bg_img_path)

                height, width, _ = bg_img.shape
                x = np.random.randint(0, width-self.crop_size[0])
                y = np.random.randint(0, height-self.crop_size[1])

                bg_img = bg_img[y:y+self.crop_size[1], 
                                x:x+self.crop_size[0], 
                                :]
                src_img = results['img']

                # stain normalize
                normalizer.fit(bg_img)
                if normalizer.stain_matrix_target is not None:
                    break

            #import pdb;pdb.set_trace()
            #save_dir = 'tests/vis_imgs'
            #image_name = os.path.split(results['filename'])[-1]
            #ori_save_path = os.path.join(save_dir, 'ori_' + image_name)
            #trans_save_path = os.path.join(save_dir, 'trans_' + image_name)
            #bg_save_path = os.path.join(save_dir, 'bg_' + image_name)
            #ori_img = src_img.copy()
            #for gt_bbox in results['gt_bboxes']:
            #    xmin, ymin, xmax, ymax = map(int, gt_bbox)
            #    cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            #cv2.imwrite(ori_save_path, ori_img)
            #cv2.imwrite(bg_save_path, bg_img)


            src_img = normalizer.transform(src_img)

            ## paste boxes region
            mask = np.zeros(self.crop_size, dtype=np.uint8)
            for gt_bbox in results['gt_bboxes']:
                xmin, ymin, xmax, ymax = map(int, gt_bbox)
                bbox_region = src_img[ymin:ymax, xmin:xmax, :]
                # bbox_region = normalizer.transform(bbox_region)

                bg_img[ymin:ymax, xmin:xmax, :] = bbox_region
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, 3)

            inpaint_img = cv2.inpaint(bg_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
            results['img'] = inpaint_img
            
            
            #for gt_bbox in results['gt_bboxes']:
            #    xmin, ymin, xmax, ymax = map(int, gt_bbox)
            #    cv2.rectangle(inpaint_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            #cv2.imwrite(trans_save_path, inpaint_img)

        return results


@PIPELINES.register_module
class RandomShiftGtBBox(object):
    def __init__(self, shift_rate=0.2, iou_threshold=0.05, overlap_threshold=0.1):
        """
            Randomly shift the gt_bboxes.
            - inputs:
                shift_rate: the probability to shift a gt bbox.
                iou_threshold: the IoU thredshold to keep the gt_bbox.
                overlap_thred: the overlap thredshold to keep the gt_bbox.
        """


        self.shift_rate = shift_rate
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        
    def __call__(self, results):

        shift_flag = np.random.rand(results['gt_bboxes'].shape[0]) < self.shift_rate

        shift_id = np.where(shift_flag)[0]
        keep_id = np.where(~shift_flag)[0]

        if len(shift_id) == 0:  # no shifting
            return results


        shift_gt_bboxes = results['gt_bboxes'][shift_id, :]
        keep_gt_bboxes = results['gt_bboxes'][keep_id, :]
        shift_gt_labels = results['gt_labels'][shift_id]
        keep_gt_labels = results['gt_labels'][keep_id]

        src_img = results['img']

        gt_bboxes = keep_gt_bboxes
        gt_labels = keep_gt_labels
        shift_gt_bbox_list = []


        mask = np.zeros((src_img.shape[0], src_img.shape[1], 1), np.uint8)
        for i in range(len(shift_gt_bboxes)):   # inpainting mask
            ltx = int(shift_gt_bboxes[i, 0])
            lty = int(shift_gt_bboxes[i, 1])
            brx = int(shift_gt_bboxes[i, 2])
            bry = int(shift_gt_bboxes[i, 3])

            mask[lty: bry, ltx: brx, :] = 1

            shift_gt_bbox_list.append(
                src_img[lty: bry, ltx: brx, :].copy()
            )

        src_img = cv2.inpaint(src_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

        h = np.min([src_img.shape[0], results['ori_shape'][0]])
        w = np.min([src_img.shape[1], results['ori_shape'][1]])


        for i in range(len(shift_gt_bbox_list)):
            copy_image = shift_gt_bbox_list[i]
            copy_image_h,copy_image_w = copy_image.shape[:2]
            for attempt in range(10):
                if copy_image_h >= h or copy_image_w >= w:
                    break
                top = np.random.randint(0,h-copy_image_h)
                left = np.random.randint(0,w-copy_image_w)
                box1 = np.array([left,top,left+copy_image_w,top+copy_image_h])
                check_overlap_flag = self._check_box_overlap(box1,gt_bboxes)
                if check_overlap_flag:
                    src_img[top:top+copy_image_h,left:left+copy_image_w,:] = copy_image
                    gt_bboxes = np.concatenate((gt_bboxes,box1[None,:]),axis=0)
                    gt_labels = np.concatenate((gt_labels,shift_gt_labels[i].reshape(1)),axis=0)
                    
                    break   


        results['img'] = src_img
        results['gt_bboxes'] = gt_bboxes.astype(results['gt_bboxes'].dtype)
        results['gt_labels'] = gt_labels

        return results


    def _check_box_overlap(self,box1,box_list):
        len_box_list = len(box_list)
        
        if len_box_list == 0:
            return True
        
        b1_x1,b1_y1,b1_x2,b1_y2 = box1
        b_x1_list,b_y1_list,b_x2_list,b_y2_list = box_list[:,0],box_list[:,1],box_list[:,2],box_list[:,3]

        inter_rect_x1 = np.maximum(b1_x1,b_x1_list)
        inter_rect_y1 = np.maximum(b1_y1,b_y1_list)
        inter_rect_x2 = np.minimum(b1_x2,b_x2_list)
        inter_rect_y2 = np.minimum(b1_y2,b_y2_list)
        inter_area = np.maximum(inter_rect_x2-inter_rect_x1, 0)*np.maximum(inter_rect_y2-inter_rect_y1, 0)
        
        box1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        box_area_list = np.maximum(b_x2_list-b_x1_list,0)*np.maximum(b_y2_list-b_y1_list,0)
        iou_list = inter_area/(box1_area+box_area_list-inter_area)
        
        box1_overlap = inter_area / box1_area
        box_list_overlap = inter_area / box_area_list
        
#         pdb.set_trace()
        # print(box1_overlap, box_list_overlap)
        
        if np.max(iou_list)>self.iou_threshold:
            return False
        if np.max(box1_overlap) > self.overlap_threshold:
            return False
        if np.max(box_list_overlap)>self.overlap_threshold:
            return False
        
        return True


@PIPELINES.register_module
class ReplaceBackgroundFromSameROI(object):
    def __init__(self, drop_rate, crop_size, ann_dir):

        self.drop_rate = drop_rate
        self.crop_size = crop_size
        self.ann_dir = ann_dir

    def load_annotation(self, ann_filepath):
        with open(ann_filepath, 'r') as f:
            ann = json.load(f)
            
            roi_x  = ann['x']
            roi_y  = ann['y']
            width  = ann['w']
            height = ann['h']

            pos_list = ann['pos_list']
            pos_nums = len(pos_list)

        gt_bboxes = np.zeros((pos_nums, 4), dtype=np.float32)
        gt_labels = np.zeros((pos_nums), dtype=np.int64)        

        for idx, pos in enumerate(pos_list):
            x1 = max(pos['x']-roi_x, 0)
            y1 = max(pos['y']-roi_y, 0)
            x2 = min(x1+pos['w'], width)
            y2 = min(y1+pos['h'], height)

            gt_bboxes[idx, :] = [x1, y1, x2, y2]
            gt_labels[idx] = 1

        return gt_bboxes, gt_labels


    def crop_patch(self, img, bboxes, labels):
        H, W, C = img.shape
        px, py = self.crop_size

        # obj_num = bboxes.shape[0]
        # index = np.random.randint(0, obj_num)
        # ux, uy, vx, vy = bboxes[index, :]
        # nx = np.random.randint(max(vx - px, 0), min(ux + 1, W - px + 1))
        # ny = np.random.randint(max(vy - py, 0), min(uy + 1, H - py + 1))

        nx = np.random.randint(0, W - px)
        ny = np.random.randint(0, H - py)

        patch_coord = np.zeros((1, 4), dtype="int")
        patch_coord[0, 0] = nx
        patch_coord[0, 1] = ny
        patch_coord[0, 2] = nx + px
        patch_coord[0, 3] = ny + py

        index = self.compute_overlap(patch_coord, bboxes, 0.5)
        index = np.squeeze(index, axis=0)

        patch = img[ny: ny + py, nx: nx + px, :]
        bboxes = bboxes[index, :]
        bboxes[:, 0] = np.maximum(bboxes[:, 0] - patch_coord[0, 0], 0)
        bboxes[:, 1] = np.maximum(bboxes[:, 1] - patch_coord[0, 1], 0)
        bboxes[:, 2] = np.minimum(self.crop_size[0] + bboxes[:, 2] - patch_coord[0, 2], self.crop_size[0])
        bboxes[:, 3] = np.minimum(self.crop_size[1] + bboxes[:, 3] - patch_coord[0, 3], self.crop_size[1])

        labels = labels[index]

        return patch, bboxes, labels
        
    def compute_overlap(self, a, b, over_threshold=0.5):
        """
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = area

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        overlap = intersection / ua
        index = overlap > over_threshold
        return index


    def __call__(self, results):

        if np.random.uniform(0,1) < self.drop_rate:

            image_path = results['filename']
            image_name = image_path.split('/')[-1]
            ann_filename = image_name.replace('png', 'json')
            ann_filepath = osp.join(self.ann_dir, ann_filename)

            roi_img = cv2.imread(image_path)
            gt_bboxes, gt_labels = self.load_annotation(ann_filepath)
            
            crop_img, gt_bboxes, gt_labels = self.crop_patch(roi_img, gt_bboxes, gt_labels)
            src_img = results['img']


            ## paste boxes region
            mask = np.zeros(self.crop_size, dtype=np.uint8)
            for bbox, label in zip(results['gt_bboxes'], results['gt_labels']):
                xmin, ymin, xmax, ymax = map(int, bbox)
                bbox_region = src_img[ymin:ymax, xmin:xmax, :]
                w = int(xmax - xmin)
                h = int(ymax - ymin)
                
                for _ in range(10):
                    paste_xmin = np.random.randint(0, self.crop_size[0] - w)
                    paste_ymin = np.random.randint(0, self.crop_size[1] - h)
                    paste_xmax = min(paste_xmin + w, self.crop_size[0])
                    paste_ymax = min(paste_ymin + h, self.crop_size[1])

                    paste_bbox = np.array([[paste_xmin, paste_ymin, paste_xmax, paste_ymax]])
                    if gt_bboxes.size == 0:
                        crop_img[paste_ymin:paste_ymax, paste_xmin:paste_xmax, :] = bbox_region
                        gt_bboxes = np.concatenate((gt_bboxes, paste_bbox), axis=0)
                        gt_labels = np.concatenate((gt_labels, np.array([label])), axis=0)
                        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, 3)
                        break

                    # import pdb;pdb.set_trace()
                    index = self.compute_overlap(paste_bbox, gt_bboxes, 0.05)
                    if index.all() == False:
                        crop_img[paste_ymin:paste_ymax, paste_xmin:paste_xmax, :] = bbox_region
                        gt_bboxes = np.concatenate((gt_bboxes, paste_bbox), axis=0)
                        gt_labels = np.concatenate((gt_labels, np.array([label])), axis=0)
                        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, 3)
                        break

            if gt_bboxes.size != 0:
                inpaint_img = cv2.inpaint(crop_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
                results['img'] = inpaint_img
                results['gt_bboxes'] = gt_bboxes.astype(results['gt_bboxes'].dtype)
                results['gt_labels'] = gt_labels

        return results
