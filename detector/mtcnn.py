########################################################
#      
#      Autor: Andrej Paníček           
#      Posledná zmena: 10.5.2019
#      Popis: Obsahuje triedu MTCNN, čo je hlavná trieda
#      mojej implementácie MTCNN.Kód je inšpirovaný 
#      originálnou implementáciou v matlabe,
#      ktorá je dostupná z https://github.com/kpzhang93/MTCNN_face_detection_alignment
#
########################################################
import sys
sys.path.insert(0, "../dataset")
import torch
import numpy as np
import matplotlib.pyplot as plt
from image import Image_processor, draw_boxes, show_image, unnormalize_image
import cv2
from process_dataset import   create_dataset_from_mtcnn_output
from nets import FirstNet, SecondNet, LastNet
from func import load_model, save_model, whats_inside
import torch.nn.functional as F
from third_party_library import non_max_suppression


def draw_only_positive(image, g_truth, boxes, score=None):

    pos, sign_coor = whats_inside(ground_truths=g_truth, rois=boxes, threshold=0.5)

    draw_all(image, boxes[pos,:], score)

def draw_all(image, boxes, score=None, class_num=None):

    show_image(draw_boxes((image-127.5)/127.5, boxes[:,0:4], class_num=class_num, score=score, blue=255, green=0, red=0))

class MTCNN:
    """
    Main detector object running whole process of detection
    """

    counter = 0

    def __init__(self,
                 min_detect_size = 12,
                 min_object_size = 20,
                 factor=0.709,
                 thresholds=[0.6, 0.6, 0.7],
                 gpu=False):
        '''
        min_detect_size - minimum size of image we wanna detect
        min_object_size - minimum size of object we wanna detect
        factor - factor of scaling
        thresholds - tresholds value for each network
        gpu - if True, will run on gpu otherwise Cpu - not implemented !!!
        '''

        self._min_detect_size = min_detect_size
        self._min_object_size = min_object_size
        self._factor = factor
        self._thresholds = thresholds
        self._gpu_device = gpu
        self.first_net = FirstNet("binary", channels=3)
        load_model(self.first_net, "../models/final_pnet.pt", mode="test")
        self.sec_net = SecondNet("binary", channels=3)
        load_model(self.sec_net, "../models/final_rnet.pt", mode="test")
        self.last_net = LastNet("multi", channels=3)
        load_model(self.last_net, "../models/final_onet.pt", mode="test")


    def detect(self, image_path, sign_coor=None, layer=None):
        '''
        image - image we wanna detect traffic sign in
        '''
        all_boxes = np.array([], dtype=np.float32)

        #create image pyramid
        image_proc = Image_processor(image_path, sign_coor)
        scale_arguments = dict()
        scale_arguments['factor'] = self._factor
        scale_arguments['min_detect_size'] = self._min_detect_size
        scale_arguments['min_object_size'] = self._min_object_size


        #################First  phase#####################
        for scaled_image in image_proc.create_pyramide(scale_arguments):

            #prepare image input and feed it into net
            image_input = self._prepare_net_input(scaled_image['image'])
            class_map, box_regression = self.first_net(image_input)
            class_map = F.softmax(class_map, dim=1)

            #create bounding boxes, using only output from classification,
            #which represent probability of sign beeing inside feature
            bounding_boxes = self._create_bounding_boxes(class_map[0][1:].data.numpy(),
                                                         box_regression[0].data.numpy(),
                                                         scaled_image['scale'],
                                                         self._thresholds[0])


            res = non_max_suppression(bounding_boxes,0.5)
            f_boxes = bounding_boxes[res]
                
            if all_boxes.shape[0] == 0 and f_boxes.shape[0] > 0:
                all_boxes = f_boxes
            elif f_boxes.shape[0] > 0:
                all_boxes = np.concatenate([all_boxes, f_boxes])

        #now with boxes created from every scale of image
        res = non_max_suppression(all_boxes,0.7)
        all_boxes = all_boxes[res]

        if len(all_boxes) <= 0:
            #if you do it different, change also bechmark.py, whick except this error
            raise ValueError("You shouldnt be here, do something about it.")

        all_boxes = self._add_box_regresion(all_boxes)
        all_boxes[:,0:4] = np.fix(self._box_to_square(all_boxes)[:,0:4])
        
        boxes_width = all_boxes[:,2] - all_boxes[:,0] + 1
        boxes_height = all_boxes[:,3] - all_boxes[:,1] + 1

        roi_boxes = self._box_offset(all_boxes,image_proc.width(), image_proc.height())

        #if layer is set to zero return output we got
        if layer == 0:
            return  roi_boxes, image_proc, boxes_width, boxes_height

        #################Second  phase#####################
        sec_input = []
        for index in range(all_boxes.shape[0]):
            empty_frame = np.zeros((int(boxes_height[index]),int(boxes_width[index]),3))
            offsets = roi_boxes['offsets'][index]

            x1 = offsets[0]
            y1 = offsets[1]
            x2 = empty_frame.shape[1] + offsets[2]
            y2 = empty_frame.shape[0] + offsets[3]
            empty_frame[int(y1):int(y2),int(x1):int(x2)] = image_proc.crop_picture(*roi_boxes['pictures'][index][0:4])
            new_image = cv2.resize(empty_frame, dsize=(24,24))
            sec_input.append(new_image)

        # feed second_net
        batch = torch.from_numpy(np.stack(sec_input)).type(torch.FloatTensor).permute(0,3,1,2)

        class_map, box_regression = self.sec_net(batch)
        class_map = F.softmax(class_map, dim=1)

        #check confidence score for sign being in image
        idx = np.where(class_map[:,1:] >= self._thresholds[1])
        #update score, and use only boxes with score higger than threhold[1]
        all_boxes[idx[0],4] = class_map[idx[0],idx[1]+1].data.numpy()
        all_boxes = all_boxes[idx[0],:]
        rg = box_regression[idx[0],:].data.numpy()

        #draw_all(image_proc.full_image(), all_boxes[:,0:4])
        if all_boxes.shape[0] > 0:
            res = non_max_suppression(all_boxes,0.7)
            all_boxes = self._add_box_regresion(all_boxes[res,], rg[res,])
            all_boxes[:,0:4] = np.fix(self._box_to_square(all_boxes)[:,0:4])
       
        if len(all_boxes) <= 0:
            #show_image(image_proc.full_image(), normalized=False)
            raise ValueError("You shouldnt be here, do something about it.")
       
        boxes_width = all_boxes[:,2] - all_boxes[:,0] + 1
        boxes_height = all_boxes[:,3] - all_boxes[:,1] + 1
       
        roi_boxes = self._box_offset(all_boxes,image_proc.width(), image_proc.height())

        if layer == 1:
            return  roi_boxes, image_proc, boxes_width, boxes_height

        #################Third  phase#####################
        third_input = []
        for index in range(all_boxes.shape[0]):
            empty_frame = np.zeros((int(boxes_height[index]),int(boxes_width[index]),3))
            offsets = roi_boxes['offsets'][index]
            x1 = offsets[0]
            y1 = offsets[1]
            x2 = empty_frame.shape[1] + offsets[2]
            y2 = empty_frame.shape[0] + offsets[3]
            empty_frame[int(y1):int(y2),int(x1):int(x2)] = image_proc.crop_picture(*roi_boxes['pictures'][index][0:4])
            new_image = cv2.resize(empty_frame, dsize=(48,48))
            third_input.append(new_image)
        
        
        batch = torch.from_numpy(np.stack(third_input)).type(torch.FloatTensor).permute(0,3,1,2)
        
        # feed third_net
        class_map, box_regression = self.last_net(batch)
        class_map = F.softmax(class_map, dim=1)
        
        # #check confidence score for sign being in image
        idx = np.where(class_map[:,1:] >= self._thresholds[2])
        #!!!!!!+1 because, we got index without background
        sign_classes = idx[1]
        # #update score, and use only boxes with score higger than threhold[1]
        all_boxes[idx[0],4] = class_map[idx[0],idx[1]+1].data.numpy()
        all_boxes = all_boxes[idx[0],:]
        
        #add class type
        rg = box_regression[idx[0],:].data.numpy()

        if all_boxes.shape[0] > 0:
            all_boxes = self._add_box_regresion(all_boxes, rg)
            res = non_max_suppression(all_boxes,0.7, "Min")
            all_boxes = np.concatenate((all_boxes[res],sign_classes[np.newaxis].T[res]), axis=1)
          

        #draw_all(image_proc.full_image(), all_boxes[:,0:4], all_boxes[:,4], all_boxes[:,5])
        return all_boxes

        



    def extract_output_samples(self, image, sign_info, layer, path, mode):
        """
        args:
            layer - choose layer from which output will be created dataset
        """
        if layer not in [0, 1]:
            raise ValueError("Wrong layer number ->", layer)

        roi_boxes, image_proc, boxes_width, boxes_height = self.detect(image, sign_info, layer=layer)

        #threshold 0 because we we wanna negatves with  0 IOU, if we chose 0.45 for parts, this algorithm whould parse
        #sample with ex. 0.1 as negatives, thats what we dont wont
        pos, sign_coor = whats_inside(ground_truths=image_proc.signs_coor(), rois=roi_boxes['pictures'][:,:4], threshold=0)
        create_dataset_from_mtcnn_output(image_proc=image_proc,
                                         b_boxes=roi_boxes,
                                         width=boxes_width,
                                         height=boxes_height,
                                         sign_position=sign_coor,
                                         size=24*(layer+1),
                                         #size=48,
                                         dataset_path=path,
                                         mode=mode,
                                         neg_delete=0)


    def _add_box_regresion(self, boxes, separate_regresion=None):
        """
        Count in box regresion
        separate_regresion - array containing values for box regresion, if is None
                values are in argument "boxes"
        """

        boxes_width = boxes[:,2] - boxes[:,0] + 1
        boxes_height = boxes[:,3] - boxes[:,1] + 1

        if separate_regresion is None:
            reg = boxes[:,5:]
        else:
            reg = separate_regresion

        x1 = boxes[:,0] + reg[:,0]*boxes_width
        y1 = boxes[:,1] + reg[:,1]*boxes_height
        x2 = boxes[:,2] + reg[:,2]*boxes_width
        y2 = boxes[:,3] + reg[:,3]*boxes_height
        kalibrated_boxes = np.array([x1, y1, x2, y2])
        score = boxes[:,4]

        return np.concatenate((kalibrated_boxes, score.reshape(1,-1)), axis=0).T

    def _create_bounding_boxes(self, class_scoore_map, box_regression, scale, threshold):
        """
        So how it works ? We got input from our first network.
        Parameter class_scoore_map represent probability of sign being inside
        feature. Lets say ouput has 20 features. Each feature represent one NxN(self._min_detect_size)
        part of image. We can easily calculate how many features picture has,
        based on image size and strides inside net. Now when we know all of this
        we can create boundin boxes. We use threshold because we don't care about
        features with low probability of finding image.
        """
        stride = 2
        indicies = np.where(class_scoore_map>=threshold)
        #no a single fire up
        if len(indicies[0]) == 0:
            return(np.array([]))

        #get regresion values for bounding boxes, but only good one
        x1 = box_regression[0, indicies[1], indicies[2]]
        y1 = box_regression[1, indicies[1], indicies[2]]
        x2 = box_regression[2, indicies[1], indicies[2]]
        y2 = box_regression[3, indicies[1], indicies[2]]

        #coordinates of bounding box in original image
        lt_x1 = np.fix((stride*indicies[2])/scale)
        lt_y1 = np.fix((stride*indicies[1])/scale)
        rb_x2 = np.fix((stride*indicies[2] + self._min_detect_size)/scale)
        rb_y2 = np.fix((stride*indicies[1] + self._min_detect_size)/scale)
        score = class_scoore_map[indicies[0],indicies[1], indicies[2]]

        #concatenate every information for each box and return it
        box_position = np.array([lt_x1, lt_y1, rb_x2, rb_y2])
        box_reg = np.array([x1, y1, x2, y2])

        return np.concatenate((box_position, score.reshape(1,-1), box_reg), axis=0).T
        

    def _box_offset(self, boxes, image_width, image_height):
        """
        get coordinates of region in picture and offsets for indecies which are
        over boundaries


        pictures - coordinates of region in image we wanna crop_imgs
        offsets - offset for each coordinates which are over bounderies
        """
        coordinates = {
            'offsets' : None,
            'pictures' : None
        }

        image_width -= 1
        image_height -= 1
        offsets = np.zeros((boxes.shape[0], 4))

        picture_crop = np.copy(boxes)

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        res = np.where(x1<0)
        picture_crop[res,0] = 0
        offsets[res,0] = - x1[res]

        res = np.where(y1<0)
        picture_crop[res,1] = 0
        offsets[res,1] = - y1[res]

        res = np.where(x2>image_width)
        picture_crop[res,2] = image_width
        offsets[res,2] = -(x2[res] - image_width)

        res = np.where(y2>image_height)
        picture_crop[res,3] = image_height
        offsets[res,3] = -(y2[res] -image_height)

        coordinates['offsets'] = offsets
        coordinates['pictures'] = picture_crop

        return coordinates

    def _box_to_square(self, boxes):
        """
        get indexes of box,  make box out of it
        """
        box_width = (boxes[:,2] - boxes[:,0]) * 0.5
        box_height = (boxes[:,3] - boxes[:,1]) * 0.5

        bigger_side = np.maximum(box_width, box_height)
        boxes[:,0] = boxes[:,0] + box_width - bigger_side
        boxes[:,1] = boxes[:,1] + box_height- bigger_side
        boxes[:,2] = boxes[:,0] + bigger_side*2
        boxes[:,3] = boxes[:,1] + bigger_side*2

        return boxes


    def _prepare_net_input(self, scaled_image):
        """
        get image of scaled vector and change it to input matrix for pytorch_input
        return
        """
        torch_matrix = torch.from_numpy(scaled_image)
        return torch_matrix.permute(2,0,1).unsqueeze(0)
