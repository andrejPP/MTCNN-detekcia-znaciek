########################################################
#      
#      Author: Andrej Paníček           
#      Last Update: 26.4.2019
#      Description: This model is used for creating datasets
#           valid for training of mtcnn model.
#           For generating samples are used two already existing
#           sign datasets, GTSRB, GTSDB
#                              
#
########################################################

import sys
import random
import cv2
import os
import csv
import random
import argparse
import numpy as np
from third_party_library import compute_IOU
from func import best_matches
from mtcnn_new import MTCNN
from datasets_wrapper import classification_dataset_wrapper, detection_dataset_wrapper
from dataset_generator import DatasetGenerator
from image import ImageProcessor, draw_boxes, image_load, change_channel_order, unnormalize_image, show_image


def cut_full(full_image, coor):
    """
    Crop part of image specified by coordinates stored in
    coor parameter.

    Args:
        full_image: imput image, crop part of this image
        coor: input coordinates-> [x1, y1, x2, y2]
    Return: 
        Croped part of the image.
    """
    x1 = coor[0]
    y1 = coor[1]
    x2 = coor[2]
    y2 = coor[3]
    return full_image[y1:y2, x1:x2]


def cut_part(sign_image, coor):
    """
    Randomly offset coordinates specified by coor,
    and crop it out.

    Args:
        sign_image: imput image, crop part of this image
        coor: input coordinates-> [x1, y1, x2, y2]
    Return: 
        cut_image: cropped part of image
        [x1,y1,x2,y2]: coordinates of cropped part in original image
    """

    width = coor[2] - coor[0] + 1
    height = coor[3] - coor[1] + 1

    horizontal_offset, vertical_offset = random_image_offset(width-1, height-1, 0.76, 0.88)

    x1 = coor[0] + int(horizontal_offset)
    y1 = coor[1] + int(vertical_offset)
    x2 = x1 + width - 1
    y2 = y1 + height - 1

    if (x1 < 0) or (y1 < 0) or (x2 > sign_image.shape[1]) or (y2 > sign_image.shape[0]):
        return None, None

    sign_coor = np.array([coor[0], coor[1], coor[2], coor[3]])
    cut_coor = np.array([x1, y1, x2, y2])

    cut_image = sign_image[y1:y2, x1:x2]

    iou = compute_IOU(sign_coor, cut_coor)
    if (iou < 0.4) or (iou > 0.65):
        print(sign_coor, file=sys.stderr)
        print(cut_coor, file=sys.stderr)
        print(iou, file=sys.stderr)
        raise ValueError("Iou out of range.")

    x1_offset = coor[0] - x1
    y1_offset = coor[1] - y1
    x2_offset = x1_offset * -1
    y2_offset = y1_offset * -1
    return cut_image, [x1_offset, y1_offset, x2_offset, y2_offset]


def random_image_offset(width, height, low, high):
    """
    Args:
        width: width of image we want to shift
        eight: height of image we want to shift
        low: lowes percentage for each side of image, we can shift
        high: highest percentage for each side of image, we can shift
    Return:
        horizontal_offset: how much shift image horizontaly
        vertical_offset: how much shift image verticaly
    """

    horizontal_random_ratio = np.round(np.random.uniform(low=low, high=high), 3)
    vertical_random_ratio = np.round(np.random.uniform(low=low, high=high), 3)

    # Randomly multiple by one of {1,-1}, for change in direction.
    horizontal_offset = width - (width*horizontal_random_ratio)
    horizontal_offset *= np.random.choice([-1, 1])

    vertical_offset = height - (height*vertical_random_ratio)
    vertical_offset *= np.random.choice([-1, 1])
    
    if horizontal_random_ratio > 0.82:
        horizontal_offset = round_from_zero(horizontal_offset)
    else:
        horizontal_offset = round_to_zero(horizontal_offset)
    if vertical_random_ratio > 0.82:
        vertical_offset = round_from_zero(vertical_offset)
    else:
        vertical_offset = round_to_zero(vertical_offset)

    return horizontal_offset, vertical_offset


def random_adjust_box(coor, image_height, image_width, mode):
    """
    Randomly adjust bounding box based on mode. 

    Args:
        coor: coordinates to be adjusted
        image_height: 
        image_width: 
        mode: either "resize" -> change size of box,
                   or "shift" -> move box coordinates in one way
    Return:
        Adjusted coordinates.
    """

    if mode not in ["resize", "shift"]:
        raise ValueError("Not available mode ->", mode)

    height_bound = image_height - 1 
    width_bound = image_width - 1
    wait_for_valid_coor = True
    new_coor = np.array(coor, copy=True)

    while wait_for_valid_coor:
        # Run this loop until all coordinates are inside image.
        if mode == "resize":
            size = np.random.randint(-width_bound/5,width_bound/5)
            new_coor[:2] = coor[:2] + size
            new_coor[2:] = coor[2:] - size
        if mode == "shift":
            shift_w = np.random.randint(-width_bound/7,width_bound/7)
            shift_h = np.random.randint(-width_bound/7,width_bound/7)
            new_coor[0] = coor[0] + shift_w
            new_coor[1] = coor[1] + shift_h
            new_coor[2] = coor[2] + shift_w
            new_coor[3] = coor[3] + shift_h

        if (new_coor[0] < 0 or new_coor[1] < 0 or
            new_coor[2] > width_bound or new_coor[3] > height_bound):
            continue
        return new_coor


def normalize_coordinates(norm_max, width, height, coor):
    """
    Args:
        norm_max: highest possible value 
        width: width of cutted image
        height: height of cutted image
        coor: coordinates [x1, y1, x2, y2]
    Return:
        np.array of normalized euclidian coordinates
    """

    # Catch possible division by zero in calculation.
    if width < 2 or height < 2:
        return np.array([0,0,0,0], dtype=np.float32)

    # Always normalize to <0,norm_max> interval.
    # Subtract 1, example: width is 45, but max value for x2 coordinate is 44
    # if we put 44 into our expression, we wont get 1 as we expect
    # but something lower.
    width_ratio = norm_max/(width - 1)
    height_ratio = norm_max/(height - 1)

    x1 = width_ratio * coor[0]
    y1 = height_ratio * coor[1]
    x2 = width_ratio * coor[2]
    y2 = height_ratio * coor[3]

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def round_to_zero(number):
    return np.fix(number)


def round_from_zero(number):
    """
    Round number to nearest integer away from 0
    example:
        -15,4 -> -16
        15,4 -> 16
    """
    abs_val = np.abs(number)
    number = np.ceil(abs_val) * (abs_val/number)
    return number


def create_randomly(dataset_info, dataset_path, dataset_gen, mode, size):
    """
    From GTSRB classification dataset generate new dataset with proper structure.

    Args:
        dataset_info: dictionary, key is name of image file  and value  is TODO
        dataset_path: path to root directory of GTSRB dataset
        dataset_gen: instance of DatasetGenerator class
        mode: either "train" or "test"
        size: dimensions for sample
    """

    distribution = {x : 0 for x in range(0,11,1)}

    for image_name, signs in dataset_info.items():

        path = os.path.join(dataset_path, image_name)
        image = image_load(path, "BGR")

        for box in signs:
            pos_flag = False
            part_flag = False

            coor = np.array(box['coordinates'], dtype=np.int32)
            base_class =  box['class']
            super_class = box['super-class']

            print("------------------------------")
            while pos_flag == False or part_flag == False:
                #first resize box and then shift it
                resized_box = random_adjust_box(coor, image.shape[0], image.shape[1], mode="resize")
                new_box = random_adjust_box(resized_box, image.shape[0], image.shape[1], mode="shift")
                
                #crop  choosen part of image, and compute offset
                crop_image = cut_full(image, new_box)
                gt_offset = coor - new_box
                iou = compute_IOU(coor, new_box)

                #normalize to new size
                norm_coor = normalize_coordinates(norm_max=size-1, 
                                                width=crop_image.shape[1], 
                                                height=crop_image.shape[0],
                                                coor= gt_offset)
                        
                crop_image = cv2.resize(crop_image, dsize=(size, size))
                new_box_width = norm_coor[2] - norm_coor[0] + 1
                new_box_height =  norm_coor[3] - norm_coor[1] + 1

                if iou > 0.65 and pos_flag == False:
                    type_name = "positives"
                    pos_flag = True
                elif iou > 0.40 and part_flag == False:
                    type_name = "parts"
                    part_flag = True
                elif iou < 0.3:
                    type_name = "negatives"
                else:
                    # Ignore images  with  0.3 < IOU < 0.4.
                    continue

                dataset_gen.save_img(image=crop_image,
                                    sample_type=type_name,
                                    base_class=base_class,
                                    super_class=super_class,
                                    coordinates=norm_coor,
                                    box_height=new_box_height,
                                    box_width=new_box_width,
                                    mode=mode)
                
                x = int(10*iou)
                distribution[x] += 1
                #new_image =draw_boxes(image, new_box[np.newaxis,])
                #show_image(new_image, normalized=False)
    print(distribution)


def create_positives(dataset_info, dataset_path, dataset_generator, mode, size, full=1 ,crop=1, unchanged=False):
    """
    Take care of the generation of "positive" and "part" samples.
    Bottom IOU boundary between "positive" sample and ground truth box is 0.65. 
    For "part" sample it is then between 0.65 and 0.40.

    Args:
        dataset_info: information about ground truth bounding boxes stored in dict
        dataset_path: path to dataset root directory
        dataset_generator: Instance of DatasetGenerator
        mode: either "test" or "train"
        size: dimensions of samples
        full: how many "positive" samples generate from single bounding box
        crop: how many "part" generate from single bounding box
    """

    for image_name, signs in dataset_info.items():

        path = os.path.join(dataset_path, image_name)
        image = image_load(path, "BGR")

        for box in signs:
            coor = [int(float(index)) for index in box['coordinates']]

            if len(coor) == 4:
                # Generate "positive" and "part" samples from single bounding box.
                # The amount is set by "full" and "crop" parameters.
                for _ in range(full):
                    generate_positive(image, coor, str(box['class']), str(box['super-class']), size, dataset_generator, mode)
                for _ in range(crop):
                    generate_parts(image, coor, size, dataset_generator, mode)

                  
def generate_positive(image, coor, class_name, super_class, size, dataset_gen, mode):
    """
    From imput "image" crop part of image specified by "coor" coordinates,
    afterwards resized it to "size" and save image along with information.
    
    Args:
        image: input image
        coor: coordinates 
        class_name:  class name for sign inside of croped image
        super_class:  name of super class for sign inside of croped image
        size: dimensions for sample
        dataset_gen: instance of DatasetGenerator class
        mode: either "train" or "test"
    """

    try:
        crop_image = cut_full(image, coor)
        crop_image = cv2.resize(crop_image, dsize=(size, size))
        dataset_gen.save_img(image=crop_image,
                            sample_type="positives",
                            base_class=class_name,
                            super_class=super_class,
                            coordinates=np.array([0, 0, 0, 0], dtype=np.float32),
                            box_height=size,
                            box_width=size,
                            mode=mode)
    except Exception as e:
        print("Error with image_name", image)
        print(e)


def generate_parts(image, coor, size, dataset_gen, mode):
    """
    Generate part samples. Part samples is defined by MTCNN authors
    as samples that have IOU with ground truth bounding box 
    starting from 0.40 up to 0.65.
    
    Args:
        image: image 
        coor: coordinates of sign in this image
        size: dimensions of part sample
        dataset_gen: instance of DatasetGenerator class
        mode: either "test" or "train"
    """


    # The cut_part function always adds random offsets to
    # coordinates. So it's possible that in the end, they are out
    # of original image. Because of that and because of 
    # the small number of signs in GTSDB, it can repeat unsuccessful
    # cut operation 10 times.
    part_image, coordinates = cut_part(image, coor)
    if part_image is None:
        for _ in range(10):
            part_image, coordinates = cut_part(image, coor)
            if part_image is not None:
                break
        if part_image is None:
            print("I get ouf of range of picture.")
            return

    # Normalize to new size.
    norm_coor = normalize_coordinates(norm_max=size-1, 
                                    width=part_image.shape[1], 
                                    height=part_image.shape[0],
                                    coor= coordinates)
            
    part_image = cv2.resize(part_image, dsize=(size, size))
    new_box_width = norm_coor[2] - norm_coor[0] + 1
    new_box_height =  norm_coor[3] - norm_coor[1] + 1
    dataset_gen.save_img(image=part_image,
                        sample_type="parts",
                        coordinates=norm_coor,
                        box_height=new_box_height,
                        box_width=new_box_width,
                        mode=mode)


def create_negatives(negatives_info: dict,
                     dataset_path: str, 
                     dataset_gen: DatasetGenerator,
                     mode: str, 
                     goal_amount: int,
                     size: int):
    """
    Generate negative samples (samples that do not overlap sign),
    from GTSDB dataset.

    Args:
        negatives_info: contains list of images name from 
            GTSDB dataset that do not contain any sign
        dataset_gen: path to root directory where to save 
        data_proc: instance of DatasetGenerator class
        mode: either "test" or "train"
        goal_amount: how many samples to create
        size: dimensions of sample
    """

    random_side = random.randint(12, 80)
    crop_width = random_side
    crop_height = random_side

    # Calculate how many times to crop from single image.
    num_samples = int(goal_amount / len(negatives_info['names']))

    for image_name in negatives_info['names']:
        #TODO maybe use function for loading
        path = os.path.join(dataset_path, image_name)
        image = cv2.imread(path)
        if image is None:
            print("Couldn't read image", path, file=sys.stderr)
            continue

        image_width = image.shape[1]
        image_height = image.shape[0]
        upper_bound_width = image_width - crop_width - 1
        upper_bound_height = image_height - crop_height - 1

        print("------Creating negatives from image:", image_name, "-------")
        for _ in range(num_samples):
            # Generate random x1 coordinate, that x1 + width < width of image
            # Generate random y1 coordinate, that y1 + height < height of image
            x1 = np.random.random_integers(low=0, high=upper_bound_width)
            y1 = np.random.random_integers(low=0, high=upper_bound_height)
            print('Random x1 coordinates:', x1)
            print('Random y1 coordinates:', y1)
            print("========================")

            # Crop and resize it to "size" parameter, before saving.
            crop_image = cut_full(image, [x1, y1, x1+crop_width, y1+crop_height])
            crop_image = cv2.resize(crop_image, dsize=(size, size))
            dataset_gen.save_img(image=crop_image,
                               sample_type="negatives",
                               coordinates=[0, 0, 0, 0],
                               box_width=0,
                               box_height=0,
                               mode=mode)


def create_from_output(train_info, test_info, dataset_path, data_proc, size):
    """
    Creates dataset from output (cutout bounding boxes from image) of single n
    neural net in MTCNN. Which net is choosed by layer parameter. 
    In parameters receive dictionaries with info about images
    that should be feed into network.

    Args:
        train_info: dictionary contains info about images that should be used
            as MTCNN input, output will be stored as training samples
            in the new dataset 
        test_info: dictionary contains info about images that should be used
            as MTCNN input, output will be stored as testing samples
            in the new dataset
        dataset_path: path to root directory of the source dataset
        data_proc: instance of DataGenerator
        size: dimension of samples
    """

    if size not in [24, 48]:
        raise ValueError("Wrong dimension size ->", size)
    # Number that indicates to mtcnn class, output of which network
    # is required. Either 0 or 1.
    layer = size / 24 - 1

    mtcnn_detector = MTCNN()

    for mode, info in {"train": train_info, "test": test_info}.items():
        if info is None:
            continue

        for image_name, signs in info.items():
            path = os.path.join(dataset_path, image_name)

            roi_boxes, image_proc, boxes_width, boxes_height = mtcnn_detector.detect(path, signs, layer=layer)
            _, coordinates = best_matches(ground_truths=image_proc.signs_coor(), rois=roi_boxes['pictures'][:,:4], threshold=0)
            store_samples_from_output(image_proc=image_proc,
                                             b_boxes=roi_boxes,
                                             width=boxes_width,
                                             height=boxes_height,
                                             sign_position=coordinates,
                                             size=size,
                                             data_proc=data_proc,
                                             mode=mode,
                                             neg_delete=0.5)

        
def store_samples_from_output(image_proc: ImageProcessor,
                                     b_boxes: dict,
                                     width: list,
                                     height: list,
                                     sign_position: dict,
                                     size: int,
                                     data_proc: DatasetGenerator,
                                     mode: str ,
                                     neg_delete: float):
    """
    Creates dataset from the output of one of three networks in MTCNN.
    Use detected bounding box as a sample.

    Args:
    image_proc: ImageProcesor instance 
    b_boxes: dictionary of detected bounding boxes, contains two keys 
        1.offsets - Nx4 ndarray, where N is number of bounding boxes 
        for each bounding box \TODO
    width: list containing width for each one of b_boxes
    height: list containing heights for each one of b_boxes
    sign_position: dictionary, key is index of box from b_boxes
    size: dimensions of saved samples
    neg_delete: interval <0,1>, percentage of negative samples that shoud not be saved 
    """
    if not isinstance(image_proc, ImageProcessor):
        raise ValueError("Parameter image_proc has to be instance of ImageProcessor class")
    if neg_delete < 0 or neg_delete > 1:
        raise ValueError("Parameter neg_delete, can have values from interval <0,1>")
    
    # Process every bounding box.
    for index in range(len(b_boxes['pictures'])):
        box_width = int(width[index]) 
        box_height = int(height[index]) 

        if box_width <= 0 or box_height <= 0:
            continue

        img_container = np.zeros((int(height[index]), int(width[index]), 3))

        offsets = b_boxes['offsets'][index]
        picture_coor = b_boxes['pictures'][index]
        x1 = offsets[0]
        y1 = offsets[1]
        x2 = img_container.shape[1] + offsets[2]
        y2 = img_container.shape[0] + offsets[3]

        # Crop from input image, unnormalize and resize to needed size.
        img_container[int(y1):int(y2), int(x1):int(x2)] = image_proc.crop_picture(*picture_coor[0:4])
        image = unnormalize_image(img_container)
        image = cv2.resize(image, dsize=(size, size))
        image = change_channel_order(image, current=ImageProcessor.channel_order, new="BGR")

        if index not in sign_position:
            # In case this box dont catch any sign save as negative.
            # If ran value is less then our percentage threshold, 
            # skip this negative bounding box.
            ran_val = random.uniform(0.0,1.0)
            if ran_val < neg_delete:
                continue
            data_proc.save_img(image=image,
                               sample_type="negatives",
                               coordinates=[0, 0, 0, 0],
                               box_width=0,
                               box_height=0,
                               mode=mode)
            continue

        sign = sign_position[index]
        norm_coor = normalize_coordinates(norm_max=size-1, 
                                        width=img_container.shape[1], 
                                        height=img_container.shape[0],
                                        coor=sign['offset'])
        new_box_width = norm_coor[2] - norm_coor[0] + 1
        new_box_height = norm_coor[3] - norm_coor[1] + 1

        # Classify what type of sample is this.
        if sign['iou'] > 0.65:
            type_name = "positives"
        elif sign['iou'] >= 0.40:
            type_name= "parts"
        elif sign['iou'] < 0.30:
            type_name= "negatives"
        else:
            # Ignore values between 0.30 and 0.40.
            continue

        data_proc.save_img(image=image,
                            sample_type=type_name,
                            base_class=sign['class'],
                            super_class=sign['super-class'],
                            coordinates=norm_coor,
                            box_width=new_box_height,
                            box_height=new_box_width,
                            mode=mode)


def coordinates_to_bb_data(coor: list) -> list:
    """
    Transform coordinates of the bounding box defined by two points into
    coordinates of one point, height and width.
    Get coordinates as -> [x1,y1,x2,y2], compute bounding box as -> [x1, y1, height, width]
    """

    x1 = coor[0]
    y1 = coor[1]
    w = coor[2] - coor[0] + 1
    h = coor[3] - coor[1] + 1
    
    return [x1, y1, h, w]


def bb_data_to_coordinates(b_box: list) -> list:
    """
    Transform coordinates of the bounding box defined by one point, width
    and height into coordinates of two points.
    Get bounding box as -> [x1,y1,height,width], compute coordinates as -> [x1, y1, x2, y2]
    """

    x1 = b_box[0]
    y1 = b_box[1]
    x2 = b_box[0] + b_box[3] - 1
    y2 = b_box[1] + b_box[2] - 1

    return [x1, y1, x2, y2]


def generate_dataset(dataset_type, dataset_path, new_dataset_path, image_size):
    """
    Generate new dataset from GTSDB (detection) or GTSRB (classification) dataset.

    Args:
        datast_type: either "classification" (GTSRB) or "detection" (GTSDB)
        dataset_path: path to root directory of original dataset
        new_dataset_path: path to root directory where to save generated dataset
        image_size: size of one dimension in pixels (generates symmetrical samples)
    """
    data_proc = DatasetGenerator(new_dataset_path)

    if dataset_type == "classification":
        train_dataset, test_dataset = classification_dataset_wrapper(dataset_path)
        create_randomly(train_dataset, os.path.join(dataset_path, "Final_Training", "Images"), data_proc, mode="train", size=image_size)
        create_randomly(test_dataset, os.path.join(dataset_path, "Final_Test", "Images"), data_proc, mode="test", size=image_size)
    elif dataset_type == "detection":
        train_dataset, valid_dataset, test_dataset = detection_dataset_wrapper(dataset_path)
        # Creating training data.
        create_positives(train_dataset[0], dataset_path, data_proc, mode="train", size=image_size, full=1, crop=1)
        create_negatives(train_dataset[1], dataset_path, data_proc, mode="train", size=image_size, goal_amount=60000)
        # Creating test data.
        create_positives(test_dataset[0], dataset_path, data_proc, mode="test", size=image_size, full=1, crop=1)
        create_negatives(valid_dataset[1], dataset_path, data_proc, mode="test", size=image_size, goal_amount=10000)
    elif dataset_type == "from_output":
        train_dataset, valid_dataset, test_dataset = detection_dataset_wrapper(dataset_path)
        create_from_output(train_dataset[0], valid_dataset[0], dataset_path, data_proc, image_size)


        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data', choices=['detection',  "classification", "from_output"], help='type of dataset')
    parser.add_argument('load', help='path to old dataset')
    parser.add_argument('new', help='path to new dataset')
    parser.add_argument('size', type=int, help='size of samles')
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.load)
    new_dataset_path = os.path.abspath(args.new)
    generate_dataset(args.data, dataset_path, new_dataset_path, args.size)