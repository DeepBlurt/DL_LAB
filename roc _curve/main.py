#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt


# gt_bbox 真值； rois：预测值；
# 格式： （ymin, xmin, ymax, xmax）
# {'gt_bbox': [[769, 264, 805, 284], [212, 721, 230, 748]],
# 'rois': [[211, 722, 229, 745], [487, 451, 514, 480], [547, 244, 563, 258]]}


def check_bb(bb):
    """
    检查输入的Bounding Box是不是真实的数据
    """
    return True if bb[0] < bb[2] and bb[1] < bb[3] else False


def bb_Area(boxA):
    """
    计算Bounding Box的面积
    """
    return (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)


def bb_IOU(boxA, boxB ):
    """

    :param boxA:
    :param boxB:
    :param threshold:
    :return:
    """
    threshold = 0.99
    inter_bb = (max(boxA[0], boxB[0]),
                max(boxA[1], boxB[1]),
                min(boxA[2], boxB[2]),
                min(boxA[3], boxB[3]))

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = bb_Area(boxA)
    boxBArea = bb_Area(boxB)

    # 原始的代码： 2018年01月23日20:13:32之前
    # if check_bb(inter_bb):
    #     interArea = bb_Area(inter_bb)
    #     iou = interArea / float(boxAArea + boxBArea - interArea)
    #     return iou
    # else:
    #     return -1

    if check_bb(inter_bb):
        interArea = bb_Area(inter_bb)
        iou = interArea / (boxAArea + boxBArea - interArea)
        # 考虑到包含关系，且面积相差较大的时候
        if interArea / min(boxAArea, boxBArea) >= threshold:
            iou = 1
    else:
        iou = -1

    return iou


def bb_IOU_new(boxA, boxB):
    """
    绘制曲线没有阈值，给你去掉了这个参数
    :param boxA:
    :param boxB:
    :param threshold:
    :return:
    """
    # A包含B
    if boxA[0] < boxB[0] and boxA[1] < boxB[1] and boxA[2] > boxB[2] and boxA[3] > boxB[3]:
        return 1
    # B包含A
    elif boxA[0] > boxB[0] and boxA[1] > boxB[1] and boxA[2] < boxB[2] and boxA[3] < boxB[3]:
        return 1
    # 然后可以计算相交的部分
    else:
        inter_bb = (max(boxA[0], boxB[0]),
                max(boxA[1], boxB[1]),
                min(boxA[2], boxB[2]),
                min(boxA[3], boxB[3]))

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = bb_Area(boxA)
    boxBArea = bb_Area(boxB)
    if check_bb(inter_bb):
        interArea = bb_Area(inter_bb)
        iou = interArea / (boxAArea + boxBArea - interArea)
    else:
        iou = -1

    return iou


def roc(json_data):
    """
    plot the roc curve
    :param json_data:
    :return:
    """
    tp = np.zeros(10, dtype=np.float)
    fp = np.zeros(10, dtype=np.float)
    fn = np.zeros(10, dtype=np.float)
    thresh_vec = np.linspace(0.2, 1, 10)
    for iteration in range(10):
        thresh = thresh_vec[iteration]
        for img_name in json_data:
            truth_box = json_data[img_name]['gt_bbox']
            pred_box = json_data[img_name]['rois']\

            flag_truth = np.zeros(len(truth_box))
            flag_pred = np.zeros(len(pred_box))
            for i in range(len(truth_box)):
                for j in range(len(pred_box)):
                    truth = truth_box[i]
                    pred = pred_box[j]
                    iou = bb_IOU_new(truth, pred)
                    # 有匹配到
                    if iou != -1:
                        flag_truth[i] = 1
                        flag_pred[j] = 1
                        if iou > thresh:
                            tp[iteration] += 1
                        else:
                            fp[iteration] += 1
            # 计算fp和fn:
            for j in range(len(pred_box)):
                if flag_pred[j] == 0:
                    fp[iteration] += 1
            for k in range(len(truth_box)):
                if flag_truth[k] == 0:
                    fn[iteration] += 1
    print(fn)
    print(fp, tp)
    precision = tp/(tp+fp)
    recall = fp/(fp+fn)
    print(precision, '\n', recall)
    plt.plot(recall, precision)
    plt.figure(2)
    plt.plot(precision, recall)
    plt.show()
    # 计算没有匹配的对


json_path = './bbox_res.json'

with open(json_path, 'r') as f:
    json_data = json.load(f)

sample = json_data["1232.jpg"]

roc(json_data)
