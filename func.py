#############################################
#                                           #
#   Author: Andrej Paníček                  #
#   Desc: This model contain help functons  #
#                                           #
#############################################

import sys
import os
import torch
import numpy as np
import time
import json
from third_party_library import compute_IOU


class Timer:

    def __init__(self):
        self._time = None

    def start(self):
        self._time = time.time()

    def round(self,decimal=6):
        res = np.round(time.time() - self._time, decimal)
        self._time = time.time()
        return res


def load_json(file_name):
    with open(file_name) as data_file:
        data = json.load(data_file)
    return data


def save_json(data, file_name, mode="w", indent=2):
    with open(file_name, mode) as out:
        json.dump(data, out, indent=indent, ensure_ascii=False)


def is_negative(ground_truths: dict, rois: list):
    """
        Find bounding boxes(rois) indexes that have no overlay with any ground
        truth bounding box thus they are false positive.
    Args:
        ground_truths: 
        rois: list of detected regions of interests
    Return:
        Indices of false positive bounding boxes.
    """

    idxs = []
    for index, roi in enumerate(rois, 0):
        first_box = roi[:4]
        captured = False

        for sign in ground_truths:
            # Calculate overlay of every ground truth bounding box
            # with current detected bounding box. 
            sec_box = np.array(sign["coordinates"]).astype(np.float32)

            iou = compute_IOU(first_box, sec_box)
            if iou > 0:
                captured = True
                break
        if not captured:
            idxs.append(index)

    return idxs


def best_matches(ground_truths, rois, threshold=0):
    """
    Find best matches between detected region of interests (bounding boxes) and
    ground truth bounding boxes.

    Args:
        ground_truths: ground truth information about signs 
        rois: regions of interests detected by network (bounding boxes)
        threshold: threshold for IOU
    Return:
        List of ROIS indicies and dictionary with result data.
    """

    result_boxes = {}
    if rois.shape[1] != 4:
        raise ValueError("Size of 1 dimension of ROI should be 4, but instead is ", rois.shape[1])

    for sign_info in ground_truths:

        # Extract info about sign.
        class_number = sign_info["class"]
        super_class = sign_info["super-class"]
        sign_box = np.array(sign_info["coordinates"]).astype(np.float32)

        box_sign_coor = find_matches(rois, sign_box, threshold)

        # Multiple boxes can match single sign, so extract only the best one.
        for box_index, data in box_sign_coor.items():
            data['gt'] = sign_box
            data['class'] = class_number
            data['super-class'] = super_class

            # Check if is this box already in result for another sign.
            if box_index not in result_boxes:
                result_boxes[box_index] = data
            else:
                # Sometimes bbox can contain more signs, store sign data with
                # highest IOU.
                if data['iou'] > result_boxes[box_index]["iou"]:
                    result_boxes[box_index] = data

    return list(result_boxes.keys()), result_boxes


def find_matches(bboxes: np.array, sign_box: np.array, iou_threshold: float):
    """
    Find bounding boxes that passes threshold overlay with sign bounding box. 

    Args:
        bboxes: ndarray of bounding boxes to compare with ground truth
        sign_box: ground truth coordinates of single sign
    Return:
        Dictionary, where key is index of bounding box and value is again dictionary
        containing two keys, "offset" has as value offset between this bounding box
        and sign bounding box, "iou" has as value IOU between these two boxes.
    """
    iou_list = np.array([], dtype=np.float32)

    # Cmpute IOU for each bbox with sign bbox
    for bbox in bboxes:
        iou = np.array([compute_IOU(bbox, sign_box)] , dtype=np.float32)
        iou_list = np.concatenate((iou_list, iou), axis=0)

    # Get indicies of bounding box with IOU higher than threshold
    idx = np.where(iou_list >= iou_threshold)
    valid_bboxes = bboxes[idx]

    offset = sign_box - valid_bboxes
    return { box_index: { "offset": np.float32(offset[list_index]), "iou": iou_list[box_index]} for list_index, box_index in enumerate(idx[0],0)}


def load_model(net_obj, path, mode):
    path = os.path.abspath(path)
    # try to open path, in case of failure return
    try:
        open(path, 'rb')
    except FileNotFoundError:
        print("Model not created->", path)
        return

    try:
        net_obj.load_state_dict(torch.load(path))
    except RuntimeError as e:
        print(net_obj, file=sys.stderr)
        raise e

    if mode == "train":
        net_obj.train()
    elif mode == "test":
        net_obj.eval()
    else:
        raise ValueError("Not allowed mode value ->", mode)


def save_model(net_obj, path):
    # save pytorch net model into file
    path = os.path.abspath(path)
    torch.save(net_obj.state_dict(), path)