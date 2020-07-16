import numpy as np

def bbox_voting_dets(dets, vote_iou_thred):
    """
        Fused those bboxes whose IoU > vote_iou_thred.

        - inputs:
            dets: np.array. (N, 5), 5 means (x1, y1, x2, y2, score)
            vote_iou_thred: float.

        - outputs:
            res_bbox: np.array. (N', 5) 
    """
    if dets.size==0:
        return dets

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    cur_bbox = []
    res_bbox = []

    while order.size > 0:
        i = order.item(0)
        # keep.append(i)
        cur_bbox.append(dets[i, :])

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds_keep = np.where(ovr <= vote_iou_thred)[0]
        inds_vote = np.where(ovr > vote_iou_thred)[0]
        # print("inds_vote", inds_vote.shape[0])

        for j in range(inds_vote.shape[0]):
            cur_bbox.append(dets[order[inds_vote[j] + 1], :])

        res_bbox.append(bbox_voting(cur_bbox))

        cur_bbox = []
        order = order[inds_keep + 1]

    res_bbox = np.array(res_bbox)

    return res_bbox


def bbox_voting(bbox_list):
    bbox_num = len(bbox_list)
    if bbox_num == 1:
        return bbox_list[0]

    bbox_arr = np.array(bbox_list)
    score_sum = np.sum(bbox_arr[:, 4])
    score_max = np.max(bbox_arr[:, 4])

    res_vote = np.zeros(5)

    w = bbox_arr[:, 4] / score_sum

    res = bbox_arr[:, :4] * w[:, np.newaxis]
    # print(res.shape)

    res_vote[:4] = np.sum(res, axis=0)
    res_vote[4] = score_max

    # print(bbox_arr, res_vote)

    return res_vote