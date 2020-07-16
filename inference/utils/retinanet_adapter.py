import numpy as np

class RetinaNetAdapter(object):
    def __init__(self, input_shape=[800,800]):
        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.ratios = np.array([0.5, 1, 2])
        
        self.std = np.array([1., 1., 1., 1.])
        self.sizes = [2 ** x for x in self.pyramid_levels]
        self.scales = np.array([4.0000, 5.0397, 6.3496])

        self.height, self.width = input_shape
        self.anchors = self.create_anchors([self.width, self.height])

    def process(self, loc_pred, cls_pred):

        # transformed_anchors = self.regress_boxes(self.anchors, loc_pred, (self.height, self.width))
        # labels, scores = np.argmax(cls_pred, axis=1), np.max(cls_pred, axis=1)
        # scores_mask = np.reshape(scores > 0.05, -1)
        # transformed_anchors = transformed_anchors[scores_mask, :]
        # x_mins, y_mins, x_maxs, y_maxs = transformed_anchors.T
        # scores = scores[scores_mask]

        # return np.concatenate((x_mins[:, np.newaxis], 
        #                        y_mins[:,np.newaxis], 
        #                        x_maxs[:,np.newaxis], 
        #                        y_maxs[:,np.newaxis], 
        #                        scores[:,np.newaxis]), 
        #                        axis=1)

        labels, scores = np.argmax(cls_pred, axis=1), np.max(cls_pred, axis=1)
        scores_mask = np.reshape(scores > 0.05, -1)
        loc_pred = loc_pred[scores_mask]
        scores = scores[scores_mask]
        transformed_anchors = self.regress_boxes(self.anchors[scores_mask], loc_pred, (self.height, self.width))
        x_mins, y_mins, x_maxs, y_maxs = transformed_anchors.T

        return np.concatenate((x_mins[:, np.newaxis], 
                               y_mins[:,np.newaxis], 
                               x_maxs[:,np.newaxis], 
                               y_maxs[:,np.newaxis], 
                               scores[:,np.newaxis]), 
                               axis=1)

    def create_anchors(self, input_shape):
        def _generate_anchors(base_size=16):
            """
            Generate anchor (reference) windows by enumerating aspect ratios X
            scales w.r.t. a reference window.
            """
            num_anchors = len(self.ratios) * len(self.scales)
            # initialize output anchors
            anchors = np.zeros((num_anchors, 4))
            # scale base_size
            anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.ratios))).T
            # compute areas of anchors
            areas = anchors[:, 2] * anchors[:, 3]
            # correct for ratios
            anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
            anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))
            # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
            anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
            anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

            return anchors

        def gen_base_anchors(base_size=16):
            w = base_size
            h = base_size

            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)

            h_ratios = np.sqrt(self.ratios)
            w_ratios = 1 / h_ratios

            ws = (w * w_ratios[:, None] * self.scales[None, :]).reshape((-1))
            hs = (h * h_ratios[:, None] * self.scales[None, :]).reshape((-1))

            # yapf: disable
            base_anchors = np.stack(
                [
                    x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
                ],
                axis=-1).round()
            # yapf: enable

            return base_anchors

        def _shift(shape, stride, anchors):
            # shift_x = (np.arange(0, shape[1]) + 0.5) * stride
            # shift_y = (np.arange(0, shape[0]) + 0.5) * stride
            shift_x = np.arange(0, shape[1]) * stride
            shift_y = np.arange(0, shape[0]) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)

            shifts = np.vstack((
                shift_x.ravel(), shift_y.ravel(),
                shift_x.ravel(), shift_y.ravel()
            )).transpose()
            a = anchors.shape[0]
            k = shifts.shape[0]

            all_anchors = (anchors.reshape((1, a, 4)) + shifts.reshape((1, k, 4)).transpose((1, 0, 2)))
            all_anchors = all_anchors.reshape((k * a, 4))

            return all_anchors

        image_shapes = [(np.array(input_shape) + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, _ in enumerate(self.pyramid_levels):
            # anchors = _generate_anchors(base_size=self.sizes[idx])
            anchors = gen_base_anchors(base_size=self.sizes[idx])
            shifted_anchors = _shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        return all_anchors


    def regress_boxes(self, boxes, deltas, max_shape=None, wh_ratio_clip=16 / 1000):
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = (boxes[:, 0] + boxes[:, 2]) * 0.5 
        ctr_y = (boxes[:, 1] + boxes[:, 3]) * 0.5

        dx = deltas[:, 0] * self.std[0]
        dy = deltas[:, 1] * self.std[1]
        dw = deltas[:, 2] * self.std[2]
        dh = deltas[:, 3] * self.std[3]
        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = dw.clip(-max_ratio, max_ratio)
        dh = dh.clip(-max_ratio, max_ratio)
        
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = np.exp(dw) * widths
        pred_h = np.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w + 0.5
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h + 0.5
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w - 0.5
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h - 0.5

        if max_shape is not None:
            pred_boxes_x1 = pred_boxes_x1.clip(0, max_shape[1] - 1)
            pred_boxes_y1 = pred_boxes_y1.clip(0, max_shape[0] - 1)
            pred_boxes_x2 = pred_boxes_x2.clip(0, max_shape[1] - 1)
            pred_boxes_y2 = pred_boxes_y2.clip(0, max_shape[0] - 1)

        pred_boxes = np.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=1)

        return pred_boxes