########################################################
#      
#      Autor: Andrej Paníček           
#      Posledná zmena: 10.4.2019
#      Popis: Tento modul obsahuje pomocné funkcie, využité
#      v iných moduloch
#
########################################################
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


def whats_inside(ground_truths, rois, threshold=0):
    #compare ground truths with rois and the matching ones
    result_boxes = {}
    if rois.shape[1] != 4:
        raise ValueError("Size of 1 dimension of ROI should be 4, but instead is ", rois.shape[1])

    # for each sign in image, get its information an chcek if any box has matched with it
    for sign_info in ground_truths:

        class_number = sign_info["class"]
        super_class = sign_info["super-class"]
        sign_box = np.array(sign_info["coordinates"]).astype(np.float32)

        #take every box and compare it with ground truth of single sign
        #get indexes of boxes containinng this sign
        box_sign_coor = adjust_sign_coor(rois, sign_box, threshold)

        #store only bigges match 
        for box_index, data in box_sign_coor.items():
            #add ground truth coordinates, class and super class of real box
            data['gt'] = sign_box
            data['class'] = class_number
            data['super-class'] = super_class

            #check if is this box already in result for another sign
            if box_index not in result_boxes:
                result_boxes[box_index] = data
            else:
                #sometimes box can catch more signs, make sure we store box result with higher IOU
                if data['iou'] > result_boxes[box_index]["iou"]:
                    result_boxes[box_index] = data

    return list(result_boxes.keys()), result_boxes


def adjust_sign_coor(containers, sign_box, iou_threshold):
    """
    both inputs have to be np.array(), float32

    return:
        iou_cont - array containing IOU for each box from "containers" with "sign_box"
        box_sign_coor  - dictionary, where key is index of box from "containers"
            and value is array of sign coordinates inside this box

    """
    # print("Container:", containers)
    # print("Sign box:", sign_box)
    iou_cont = np.array([], dtype=np.float32)

    #compute IOU for each bounding box with sign ground thruth 
    for cont in containers:
        new_iou = np.array([compute_IOU(cont, sign_box)] , dtype=np.float32)
        iou_cont = np.concatenate((iou_cont, new_iou), axis=0)

    #get indicies of bounding box with IOU higher than threshold, and separate them into other array
    idx = np.where(iou_cont >= iou_threshold)
    valid_containers = containers[idx]

    #also save offset from ground truth box
    offset = sign_box - valid_containers

    #create dict containing coordinates for sign inside box, dict key is index of box from input boxes
    box_sign_coor = { box_index: { "offset": np.float32(offset[list_index]), "iou": iou_cont[box_index]} for list_index, box_index in enumerate(idx[0],0)}

    return box_sign_coor



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
