########################################################
#      
#      Author: Andrej Paníček           
#      Last Update: 28.4.2019
#      Description: Implementation of MTCNN model
#           in pytorch, code is inspired by original
#           matlab cone.                   
#      Content: MTCNN class, main class of MTCNN model,
#           and detect(image_path) method is used for,
#           detection 
#
########################################################
import torch
import numpy as np
import matplotlib.pyplot as plt
from image import ImageProcessor, draw_boxes, show_image, unnormalize_image
import cv2
from nets import FirstNet, SecondNet, LastNet
from func import load_model, save_model, is_negative, best_matches
import sys
import torch.nn.functional as F
from third_party_library import non_max_suppression


def draw_only_positive(image, g_truth, boxes, score=None):
    pos, sign_coor = best_matches(ground_truths=g_truth, rois=boxes, threshold=0.5)
    draw_all(image, boxes[pos,:], score)


def draw_all(image, boxes, score=None, class_num=None):
    show_image(draw_boxes((image-127.5)/127.5, boxes[:,0:4], class_num=class_num, score=score, blue=255, green=0, red=0))


class MTCNN:
    """
    Class representing MTCNN detector and his structure. 

    Args:
        min_object_size: minimum size of object we want to be able to detect
        factor: factor of scaling
        thresholds: tresholds value for each network
        gpu: if True, will try to run evaluation on Gpu otherwise use Cpu
    """

    counter = 0

    def __init__(self,
                 min_detect_size: int = 12,
                 min_object_size: int = 20,
                 factor: float = 0.709,
                 thresholds: list = [0.6, 0.6, 0.7],
                 gpu: bool = False):

        self._min_detect_size = min_detect_size
        self._min_object_size = min_object_size
        self._factor = factor
        self._thresholds = thresholds
        self._gpu_device = gpu
        self.first_net = FirstNet("binary", channels=3)
        load_model(self.first_net, "./models/final_pnet.pt", mode="test")
        self.sec_net = SecondNet("binary", channels=3)
        load_model(self.sec_net, "./models/final_rnet.pt", mode="test")
        self.last_net = LastNet("multi", channels=3)
        load_model(self.last_net, "./models/final_onet.pt", mode="test")

    def detect(self, 
               image_path: str,
               sign_coor: list = None,
               layer: int = None):
        '''
        Run detection process of sign in image.

        Args:
            image_path: path to the image that is subjected to detection 
            sign_coor: list of coordinates of signs in image, 
                ! used only in training or testing 
            layer: output of which network in cascade is interesting for us,
                possible values are 1,2,3 for training or testing and None for
                real detection 
        Return:
            If argument layer is set to 1,2 or 3 method return multiple Ndarrays
            containing advanced information about detection. If is layer None
            return one Ndarray containing bounding boxes from last net (Onet).
        '''

        # Load image and
        image_proc = ImageProcessor(image_path, sign_coor)

        # Run first phase 
        all_boxes = self.pnet_phase(image_proc) 
        if len(all_boxes) <= 0:
            # If you do this differently, changes are needed in bechmark.py.
            raise ValueError("No sign detected by PNet.")

        boxes_width = all_boxes[:,2] - all_boxes[:,0] + 1
        boxes_height = all_boxes[:,3] - all_boxes[:,1] + 1
        roi_boxes = self._box_offset(all_boxes, image_proc.width(), image_proc.height())

        # Only used for dataset creation from first phase.
        if layer == 0:
            return  roi_boxes, image_proc, boxes_width, boxes_height

        # Prepare input for second phase.
        sec_input = self._cut_rois(image_proc, 
                                   all_boxes.shape[0], 
                                   roi_boxes, 
                                   boxes_height, 
                                   boxes_width,
                                   24)
        # Run second phase.
        all_boxes = self.rnet_phase(sec_input, all_boxes)
        if len(all_boxes) <= 0:
            raise ValueError("No sign detected by RNet.")
       
        boxes_width = all_boxes[:,2] - all_boxes[:,0] + 1
        boxes_height = all_boxes[:,3] - all_boxes[:,1] + 1
        roi_boxes = self._box_offset(all_boxes, image_proc.width(), image_proc.height())

        # Only used for dataset creation from second phase.
        if layer == 1:
            return  roi_boxes, image_proc, boxes_width, boxes_height

        # Prepare input for third phase  
        third_input = self._cut_rois(image_proc, 
                                     all_boxes.shape[0], 
                                     roi_boxes, 
                                     boxes_height, 
                                     boxes_width,
                                     48)
        # Run final phase
        all_boxes = self.onet_phase(third_input, all_boxes)
        if len(all_boxes) <= 0:
            raise ValueError("No sign detected by Onet.")

        # Only used for dataset creation from last phase.
        if layer == 2:
            roi_boxes = self._box_offset(all_boxes, image_proc.width(), image_proc.height())
            return  roi_boxes, image_proc, boxes_width, boxes_height

        return all_boxes

    def pnet_phase(self,
                   image_proc: ImageProcessor) -> np.array:
        """
        From original image create a pyramid with different resolutions and 
        use Pnet to detect traffic signs in every pyramid layer. 

        Args:
            image_proc: ImageProcessor instance 
        Return:
            Ndarray containing interesting bounding boxes detected by Pnet.
        """

        all_boxes = np.array([], dtype=np.float32)
        scale_arguments = {}
        scale_arguments['factor'] = self._factor
        scale_arguments['min_detect_size'] = self._min_detect_size
        scale_arguments['min_object_size'] = self._min_object_size

        for scaled_image in image_proc.create_pyramide(scale_arguments):
            # Run detection on scaled image.
            class_map, box_reg = self._run_model(self.first_net, scaled_image["image"], pnet=True)

            # Create bounding boxes, using only output from classification,
            # which tell us probability of sign beeing inside each cell in image grid.
            bounding_boxes = self._create_bounding_boxes(class_map[0][1:].data.numpy(),
                                                         box_reg[0].data.numpy(),
                                                         scaled_image['scale'],
                                                         self._thresholds[0])
            # Extract only interesting bounding boxes from this scale.
            suitable_bboxes = bounding_boxes[ non_max_suppression(bounding_boxes, 0.5) ]
            if all_boxes.shape[0] == 0 and suitable_bboxes.shape[0] > 0:
                all_boxes = suitable_bboxes
            elif suitable_bboxes.shape[0] > 0:
                all_boxes = np.concatenate([all_boxes, suitable_bboxes])

        # Extract only interesting bounding boxes from all pyramid scales. 
        if all_boxes.shape[0] > 0:
            all_boxes = all_boxes[ non_max_suppression(all_boxes, 0.7) ]
            all_boxes = self._add_box_regression(all_boxes)
            all_boxes[:, 0:4] = np.fix(self._box_to_square(all_boxes)[:, 0:4])

        return all_boxes

    def rnet_phase(self,
                   net_input: list,
                   bboxes: np.array) -> np.array:
        """
        Get list of image regions from original image (bounding box regions
        detected in previous phase) and run another classification with Rnet
        to update confidence score for each one of them.

        Args:
            net_input: list of images (detected bounding boxes in previous phase)
            bboxes: Ndarray of bounding boxes detected in previous phase
        Return:
            Ndarray containing interesting bounding boxes detected by Rnet
        """

        # Run detection with Rnet.
        class_map, box_reg = self._run_model(self.sec_net, net_input)

        # Update confidence score for each bounding box and eliminate 
        # ones that don't pass the threshold. 
        idx = np.where(class_map[:, 1:] >= self._thresholds[1])
        bboxes[idx[0], 4] = class_map[idx[0], idx[1]+1].data.numpy()
        bboxes = bboxes[idx[0], :]
        box_reg = box_reg[idx[0], :].data.numpy()

        if bboxes.shape[0] > 0:
            # Apply bounding box regression.
            res = non_max_suppression(bboxes, 0.7)
            bboxes = self._add_box_regression(bboxes[res,], box_reg[res,])
            bboxes[:, 0:4] = np.fix(self._box_to_square(bboxes)[:, 0:4])
        
        return bboxes

    def onet_phase(self, 
                   net_input: list,
                   bboxes: np.array) -> np.array:
        """
        Run final classification with Onet on images (regions in original image)
        that previous layers confidently classified as traffic sign.

        Args:
            net_input: list of images (detected bounding boxes in previous phase)
            bboxes: Ndarray of bounding boxes detected in previous phase
        Return:
            Ndarray containing interesting bounding boxes detected by Rnet
        """

        # Run detection with Onet
        class_map, box_reg = self._run_model(self.last_net, net_input)
        
        # Find most confident class index for each bounding box.
        classes_idx = np.argmax(class_map.data.numpy(), axis=1)

        # Extract index of bounding boxes whose most confident class score
        # passes the required threshold and it isn't background class.
        # Store index of suitable bounding boxes along with detected class.
        sign_idx = [[],[]]
        for index, class_idx in enumerate(classes_idx):
            if class_idx != 0 and \
                    class_map[index, class_idx].item() > self._thresholds[2]:
                sign_idx[0].append(index)
                sign_idx[1].append(class_idx)
        
        # Update score of suitable bounding boxes for detected class.
        bboxes[sign_idx[0], 4] = class_map[sign_idx[0], sign_idx[1]].data.numpy()
        bboxes = bboxes[sign_idx[0], :]
        box_reg = box_reg[sign_idx[0], :].data.numpy()

        # Have to decrement detected classes cause 0 represented background.
        sign_classes = np.array(sign_idx[1]) -1

        if bboxes.shape[0] > 0:
            bboxes = self._add_box_regression(bboxes, box_reg)
            res = non_max_suppression(bboxes, 0.7, "Min")
            bboxes = np.concatenate((bboxes[res], sign_classes[np.newaxis].T[res]), axis=1)

        return bboxes

    def _cut_rois(self, image_proc, num_bboxes, roi_boxes, bboxes_height, bboxes_width, dim):
        """
        This method is used after first and second stage to cut updated 
        bounding boxes from original image for next stage. 
        It needs to be considered that some bounding boxes will be partly 
        outside of the image. That's why are used calculated offsets, so
        the bounding box dimensions persist.
        Also resize cropped ROIs to specified size.

        Args:
            image_proc: instance of ImageProcessor
            num_bboxes: number of bounding boxes
            roi_boxes: dictionary contains two keys
                "offsets" - offsets for bboxes that are out of original image
                "pictures" - coordinates of ROIs (regions to cut out)
            bboxes_height: Ndarray of height for each bounding box 
            bboxes_width: Ndarray of width for each bounding box
            dim: dimension of cut out
        Return:
            List of ROIs images.
        """

        rois_images = []
        for index in range(num_bboxes):
            empty_frame = np.zeros((int(bboxes_height[index]), int(bboxes_width[index]), 3))
            offsets = roi_boxes['offsets'][index]
            x1 = offsets[0]
            y1 = offsets[1]
            x2 = empty_frame.shape[1] + offsets[2]
            y2 = empty_frame.shape[0] + offsets[3]
            empty_frame[int(y1):int(y2), int(x1):int(x2)] = image_proc.crop_picture(*roi_boxes['pictures'][index][0:4])
            rois_images.append(cv2.resize(empty_frame, dsize=(dim, dim)))

        return rois_images

    def _run_model(self, net, input, pnet=False):
        """
        Run forward pass on one of the neural networks.

        Args:
            net: one of Pnet, Rnet, Onet
            input: either single image for Pnet or batch of small image (Roi)
                for Rnet and Onet
            pnet: true for Pnet otherwise false
        Return:
            Classification map ndarray and bounding box regression ndarray.
        """

        if pnet:
            # Input is single image.
            prepared_input = torch.from_numpy(input).permute(2,0,1).unsqueeze(0)
        else:
            # Input is batch of cropped images (specified by bounding boxes)
            # from original image.
            prepared_input = torch.from_numpy(np.stack(input)).type(torch.FloatTensor).permute(0,3,1,2)

        class_map, box_regression = net(prepared_input)
        class_map = F.softmax(class_map, dim=1)
        return class_map, box_regression

    
    def _add_box_regression(self, boxes, separate_regresion=None):
        """
        Adjust bounding boxes with regression values. In first phase (Pnet) 
        regresion values are part of the "boxes" argument. In second and third 
        phase (Rnet, Onet) are stored in "separate_regresion". 

        Args:
            boxes: ndarray contains bounding boxes and regression values
                in first phase
            separate_regresion: ndarray of regression values 
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

    def _create_bounding_boxes(self, 
                               class_scoore_map: np.array,
                               box_regression: np.array,
                               scale: float,
                               threshold: float) -> np.array:
        """
        This method is use in first phase on Pnet output 
        and calculates proper dimensions for interesting bounding boxes.
        Interesting bounding box is one which has one class 
        confidence score higher than required threshold.
        Dimensions of this bounding box is then calculated based
        on used scale during detection.
        In this context, single bounding box can be described as
        one destination of sliding widow on single scaled image.
        Method dont apply obtained bounding box regression.  

        Args:
            class_score_map: classification score for each cell
            box_regression: box regression for each cell
            scale: scale used in pyramide, 
                needed for calculation of correct bounding box size
            threshold: confidence value that is low boundary for bounding boxes 
        """
        stride = 2
        indicies = np.where(class_scoore_map>=threshold)
        #no a single fire up
        if len(indicies[0]) == 0:
            return(np.array([]))

        # Extract regresion values for bounding boxes.
        x1 = box_regression[0, indicies[1], indicies[2]]
        y1 = box_regression[1, indicies[1], indicies[2]]
        x2 = box_regression[2, indicies[1], indicies[2]]
        y2 = box_regression[3, indicies[1], indicies[2]]

        # Coordinates of bounding box in original image
        lt_x1 = np.fix((stride*indicies[2])/scale)
        lt_y1 = np.fix((stride*indicies[1])/scale)
        rb_x2 = np.fix((stride*indicies[2] + self._min_detect_size)/scale)
        rb_y2 = np.fix((stride*indicies[1] + self._min_detect_size)/scale)
        score = class_scoore_map[indicies[0],indicies[1], indicies[2]]

        # Concatenate informations for each bounding box
        box_position = np.array([lt_x1, lt_y1, rb_x2, rb_y2])
        box_reg = np.array([x1, y1, x2, y2])
        return np.concatenate((box_position, score.reshape(1,-1), box_reg), axis=0).T
        

    def _box_offset(self, boxes, image_width, image_height):
        """
        For bounding boxes that are exceeding image border, calculate 
        offset (how much are they out of image for each coordinate).
        Also for these bounding boxes calculate valid coordinates
        (part of the bounding box that is overlaping image).

        Args:
            boxes: ndarray of bounding boxes
            image_width: width of image that is subject of detection
            image_height: height of image that is subject of detection
        Return:
            Two keys dictionary with the following meaning,
                pictures - coordinates of regions in image to be cut out
                offsets - offsets for each bounding box exceeding image border
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
        Adjust bounding box dimensions to create square.
        
        Args:
            boxes: Ndarray of bounding boxes
        Return:
            Ndarray of all bounding boxes adjusted to square.

        """
        box_width = (boxes[:,2] - boxes[:,0]) * 0.5
        box_height = (boxes[:,3] - boxes[:,1]) * 0.5

        bigger_side = np.maximum(box_width, box_height)
        boxes[:,0] = boxes[:,0] + box_width - bigger_side
        boxes[:,1] = boxes[:,1] + box_height- bigger_side
        boxes[:,2] = boxes[:,0] + bigger_side*2
        boxes[:,3] = boxes[:,1] + bigger_side*2
        return boxes