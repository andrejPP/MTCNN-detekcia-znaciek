########################################################
#      
#      Autor: Andrej Paníček           
#      Posledná zmena: 26.4.2019
#      Popis: Tento modul obsahuje funkcie
#      využité pre generovanie dátových sád z GTSRB 
#      a GTSDB
#                              
#
########################################################

import sys
import random
import cv2
import os
import csv
import random
from third_party_library import compute_IOU
import argparse
import numpy as np
from dataset_generator import DatasetGenerator
from image import Image_processor, draw_boxes, image_load, change_channel_order, unnormalize_image, show_image


def cut_full(full_image, coor):
    """
    Crop part of image specified by coor arguments
    args:
        full_image - imput image, crop part of this image
        coor - input coordinates-> [x1, y1, x2, y2]
    return: 
        - croped part of image
    """
    x1 = coor[0]
    y1 = coor[1]
    x2 = coor[2]
    y2 = coor[3]
    return full_image[y1:y2, x1:x2]


def cut_part(sign_image, coor):
    """
    Randomly offset coordinates(not totaly random it is in range) specified by coor,
    and crop it out.
    args:
        sign_image - imput image, crop part of this image
        coor - input coordinates-> [x1, y1, x2, y2]
    return: 
        cut_image - croped part of image
        [x1,y1,x2,y2] - coordinates of cropped part, in original image
    """

    width = coor[2] - coor[0] + 1
    height = coor[3] - coor[1] + 1

    horizontal_offset, vertical_offset = random_image_offset(width-1, height-1, 0.76, 0.88)

    x1 = coor[0] + int(horizontal_offset)
    y1 = coor[1] + int(vertical_offset)
    x2 = x1 + width - 1
    y2 = y1 + height - 1

    if (x1 < 0) or (y1 < 0) or \
            (x2 > sign_image.shape[1]) or (y2 > sign_image.shape[0]):
        return None, None

    sign_coor = np.array([coor[0], coor[1], coor[2], coor[3]])
    cut_coor = np.array([x1, y1, x2, y2])

    cut_image = sign_image[y1:y2, x1:x2]

    iou = compute_IOU(sign_coor, cut_coor)
    if (iou < 0.4) or (iou > 0.65):
        print(sign_coor)
        print(cut_coor)
        print(iou)
        raise ValueError("Iou out of range")
    # print("Iou  value is:", iou)

    # get coordinates
    x1_offset = coor[0] - x1
    y1_offset = coor[1] - y1
    x2_offset = x1_offset * -1
    y2_offset = y1_offset * -1

    return cut_image, [x1_offset, y1_offset, x2_offset, y2_offset]


def random_image_offset(width, height, low, high):
    """
    args:
        width - width of image we want to shift
        height - height of image we want to shift
        low - lowes percentage for each side of image, we can shift
        high - highest percentage for each side of image, we can shift
    return:
        horizontal_offset - how much shift image horizontaly
        vertical_offset - how much shift image  verticaly
    """

    horizontal_random_ratio = np.round(np.random.uniform(low=low, high=high), 3)
    vertical_random_ratio = np.round(np.random.uniform(low=low, high=high), 3)

    #randomly multiple by one of {1,-1}, for change of direction
    horizontal_offset = width - (width*horizontal_random_ratio)
    horizontal_offset *= np.random.choice([-1, 1])

    vertical_offset = height - (height*vertical_random_ratio)
    vertical_offset *= np.random.choice([-1, 1])
    
    #round offsets to integer
    if horizontal_random_ratio > 0.82:
        horizontal_offset = round_from_zero(horizontal_offset)
    else:
        horizontal_offset = round_to_zero(horizontal_offset)

    if vertical_random_ratio > 0.82:
        vertical_offset = round_from_zero(vertical_offset)
    else:
        vertical_offset = round_to_zero(vertical_offset)

    return horizontal_offset, vertical_offset


def random_adjust_box(sign_coor, image_height, image_width, mode):
    """
    From coordinates of box in "sign_coor" create new one, based on 
    mode and height, width of image. There are only two available mods,
    resize -> change size of box
    shift -> move box coordinates in one way
    """

    if mode not in ["resize", "shift"]:
        raise ValueError("Not available mode ->", mode)

    height_bound = image_height - 1 
    width_bound = image_width - 1
    #new coordinates are inside image
    wait_for_valid_coor = True
    new_coor = np.array(sign_coor, copy=True)

    while wait_for_valid_coor:
        #change size of box
        if mode == "resize":
            size = np.random.randint(-width_bound/5,width_bound/5)
            new_coor[:2] = sign_coor[:2] + size
            new_coor[2:] = sign_coor[2:] - size
        if mode == "shift":
            shift_w = np.random.randint(-width_bound/7,width_bound/7)
            shift_h = np.random.randint(-width_bound/7,width_bound/7)
            new_coor[0] = sign_coor[0] + shift_w
            new_coor[1] = sign_coor[1] + shift_h
            new_coor[2] = sign_coor[2] + shift_w
            new_coor[3] = sign_coor[3] + shift_h

        if (new_coor[0] < 0 or new_coor[1] < 0 or
            new_coor[2] > width_bound or new_coor[3] > height_bound):
            continue
        return new_coor


def normalize_coordinates(norm_max, width, height, coor):
    """
    width  - represent width of cutted image
    height - represent height of cutted image
    coor - coordinates [x1,y1, x2,y2]

    return np.array of normalized euclidian coordinates
    """

    #after we subrtract 1 from 1 we got zero, and we cant devide by zero in next command 
    if width < 2 or height < 2:
        return np.array([0,0,0,0], dtype=np.float32)


    # always normalize to <0,norm_max> interval
    # subtract 1, example: width is 45, but max value for x2 coordinate is 44
    # if we put 44 into our expression, we wont get 1 as we expect
    # but something lower
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
                    #this is for images  w  0.3 < IOU < 0.4
                    continue

                #save image
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

def create_positives(dataset_info, dataset_path, dataset_generator, mode, size, full=1 ,crop=1, w_background=0, unchanged=False):


    for image_name, signs in dataset_info.items():

        path = os.path.join(dataset_path, image_name)
        image = image_load(path, "BGR")

        for box in signs:
            name = box['class']
            coor = [int(float(index)) for index in box['coordinates']]

            if len(coor) == 4:
                #firstly generate positive
                for _ in range(full):
                    generate_positive(image, coor, str(box['class']), str(box['super-class']), size, dataset_generator, mode)

                #create positive with more background if is w_background > 0
                for _ in range(w_background):
                    generate_positive_w_background(image, coor, str(box['class']), str(box['super-class']), size, unchanged, dataset_generator, mode)

                for _ in range(crop):
                    generate_parts(image, coor, size, dataset_generator, mode)
                  
def generate_positive(image, coor, class_name, super_class, size, dataset_gen, mode):
    """
    From imput "image"  crop part of image specified by "coor" coordinates, afterwards resized it to "size" 
    save information through "dataset_gen".
    args:
        image - imput image
        coor - coordinates 
        class_name - class name for sign inside of croped image
        super_class - name of super class for sign inside of croped image
        size - size for resize
        dataset_gen - instance of DatasetGenerator, for saving croped image
        mode - training or testing data
    
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

def generate_positive_w_background(image, coor, class_name, super_class, size, unchanged, dataset_gen, mode):

    #if unchanged is True, save image as original
    if unchanged:
        img_w_background  = image
        coordinates = coor
    else:
        img_w_background, coordinates = cut_bigger_area(image, coor, low_perc=1, up_perc=20)

    #normalize to new size
    norm_coor = normalize_coordinates(norm_max=size-1, 
                                    width=img_w_background.shape[1], 
                                    height=img_w_background.shape[0],
                                    coor=coordinates)

    img_w_background = cv2.resize(img_w_background, dsize=(size, size))
    new_box_width = norm_coor[2] - norm_coor[0] + 1
    new_box_height =  norm_coor[3] - norm_coor[1] + 1
    dataset_gen.save_img(image=img_w_background,
                        sample_type="positives",
                        base_class=class_name,
                        super_class=super_class,
                        coordinates=norm_coor,
                        box_height= new_box_height,
                        box_width= new_box_width,
                        mode=mode)

def generate_parts(image, coor,  size, dataset_gen, mode):

    # 3.from cropp part of sign and save it
    part_image, coordinates = cut_part(image, coor)
    if part_image is None:
        # unsuccesfull try, try another 5 more time, then
        for _ in range(10):
            part_image, coordinates = cut_part(image, coor)
            # if succesfull
            if part_image is not None:
                break
        # if unsuccesfull
        if part_image is None:
            print("I get ouf of range of picture.")
            return

    #normalize to new size
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


def create_negatives(negatives_info, dataset_path, data_proc, mode, goal_amount, size):

    random_side = random.randint(12, 80)
    crop_width = random_side
    crop_height = random_side

    # how many crops from one image
    num_samples = int(goal_amount / len(negatives_info['names']))

    for image_name in negatives_info['names']:
        path = os.path.join(dataset_path, image_name)
        image = cv2.imread(path)
        # some problem while reading
        if image is None:
            print("Couldn't read image",path, file=sys.stderr)
            continue

        image_width = image.shape[1]
        image_height = image.shape[0]
        upper_bound_width = image_width - crop_width - 1
        upper_bound_height = image_height - crop_height - 1

        print("------Creating negatives from image:", image_name, "-------")
        # create 5 random crops from each image
        for _ in range(num_samples):

            # generate random x1 coor, that x1 + width < width of image
            x1 = np.random.random_integers(low=0, high=upper_bound_width)
            print('Random x1 coordinates:', x1)

            # generate random y1 coor, that y1 + height < height of image
            y1 = np.random.random_integers(low=0, high=upper_bound_height)
            print('Random y1 coordinates:', y1)
            print("========================")

            # crop and resize it to 12x12
            crop_image = cut_full(
                image, [x1, y1, x1+crop_width, y1+crop_height])
            crop_image = cv2.resize(crop_image, dsize=(size, size))
            # save it as negative
            data_proc.save_img(image=crop_image,
                               sample_type="negatives",
                               coordinates=[0, 0, 0, 0],
                               box_width=0,
                               box_height=0,
                               mode=mode)



def create_dataset_from_mtcnn_output(image_proc,
                                     b_boxes,
                                     width,
                                     height,
                                     sign_position,
                                     size,
                                     dataset_path,
                                     mode,
                                     neg_delete):
    # This function works with Image_processor object from module image
    # image - image_processor object
    # boxes - net output, this function create dataset from it,, its dictionary
    #     contains two keys: 1.offsets - Nx4 ndarray, where N is number of bounding boxes,
    #     for each bounding box \TODO
    # width - array containing width for each one of b_boxes
    # height - array containing heights for each one of b_boxes
    # sign_position - dictionary, key is index of box from b_boxes
    # size - image size, we want to save
    # neg_per - interval <0,1>, how many percent of negative bounding boxes should not be saved,
    #               wher 0 is 0 percent and  1 is 100 percent
    if not isinstance(image_proc, Image_processor):
        raise ValueError(
            "image_proc argument has to be instance of Image_processor class")
    #check interval range of neg_delete, <0,1>,
    if neg_delete < 0 or neg_delete > 1:
        raise ValueError("Parameter neg_delete, can have values from interval <0,1>")
    
    data_proc = DatasetGenerator(dataset_path)
    # for each bounding box
    for index in range(len(b_boxes['pictures'])):
        box_width = int(width[index]) 
        box_height = int(height[index]) 

        #!!!Carefull, if width or height is negative, skip it
        if box_width <= 0 or box_height <= 0:
            continue

        img_container = np.zeros((int(height[index]), int(width[index]), 3))

        offsets = b_boxes['offsets'][index]
        picture_coor = b_boxes['pictures'][index]
        x1 = offsets[0]
        y1 = offsets[1]
        x2 = img_container.shape[1] + offsets[2]
        y2 = img_container.shape[0] + offsets[3]

        # crop from input image, unnormalize and resize to needed size
        img_container[int(y1):int(y2), int(x1):int(x2)] = image_proc.crop_picture(*picture_coor[0:4])
        image = unnormalize_image(img_container)
        image = cv2.resize(image, dsize=(size, size))

        #change order of color in image before saving, cv saves BGR 
        image = change_channel_order(image, current=Image_processor.channel_order, new="BGR")

        # if this box dont catche any sign save as negative
        if index not in sign_position:
            ran_val = random.uniform(0.0,1.0)
            #if ran value is less then our percentage threshold, skip negative 
            if ran_val < neg_delete:
                continue

            data_proc.save_img(image=image,
                               sample_type="negatives",
                               coordinates=[0, 0, 0, 0],
                               box_width=0,
                               box_height=0,
                               mode=mode)
            continue
        # else check what type of image we are saving
        sign = sign_position[index]

        norm_coor = normalize_coordinates(norm_max=size-1, 
                                        width=img_container.shape[1], 
                                        height=img_container.shape[0],
                                        coor=sign['offset'])

        new_box_width = norm_coor[2] - norm_coor[0] + 1
        new_box_height = norm_coor[3] - norm_coor[1] + 1

        # positive images
        if sign['iou'] > 0.65:
            type_name = "positives"
        #part images
        elif sign['iou'] >= 0.40:
            type_name= "parts"
        #negative images
        elif sign['iou'] < 0.30:
            type_name= "negatives"
        else:
            #if between 0.30 and 0.40, dont save
            continue

        data_proc.save_img(image=image,
                            sample_type=type_name,
                            base_class=sign['class'],
                            super_class=sign['super-class'],
                            coordinates=norm_coor,
                            box_width=new_box_height,
                            box_height=new_box_width,
                            mode=mode)
    


##########################FUNCTIONS FOR GERMAN DATASET###########################

def class_dataset_wrapper(dataset_path):

    train_dataset_dict = {}
    test_dataset_dict = {}
    path = os.path.join(dataset_path, "Final_Training", "Images")
    for _, dirs, _ in os.walk(path):
        for class_dir in dirs:
            info_file = os.path.join(
                path, class_dir, "GT-{}.csv".format(class_dir))

            with open(info_file, 'r') as data:
                for each in data.readlines():
                    arr = each.split(';')
                    if arr[0] == "Filename":
                        continue
                    name = os.path.join(class_dir, arr[0])
                    coor = arr[3:7]
                    base_class = arr[7].rstrip()
                    sample_info = [name, *coor, base_class,
                                   germ_super_class(int(base_class))]
                    parse_sample_information(sample_info, train_dataset_dict)

    path = os.path.join(dataset_path, "Final_Test", "Images")
    info_file = os.path.join(path, "GT-final_test.csv")
    with open(info_file, 'r') as data:
        for each in data.readlines():
            arr = each.split(';')
            if arr[0] == "Filename":
                continue
            name = arr[0]
            coor = arr[3:7]
            base_class = arr[7].rstrip()
            sample_info = [name, *coor, base_class,
                           germ_super_class(int(base_class))]
            parse_sample_information(sample_info, test_dataset_dict)

    return train_dataset_dict, test_dataset_dict


def german_dataset_wrapper(path):

    info_file = os.path.join(path, "gt.txt")
    train_dict = dict()
    valid_dict = dict()
    test_dict = dict()
    train_negative_img = [number for number in range(400)]
    valid_negative_img = [number for number in range(400,600)]
    test_negative_img = [number for number in range(600, 900)]

    with open(info_file, 'r') as data_file:
        for each in data_file.readlines():
            arr = each.split(';')
 
            arr[5] = arr[5].rstrip()
            arr.append(germ_super_class(int(arr[5])))
            if int(arr[0][:5]) < 400:
                #if is this picture in info file, it does have at least 1 sign
                #so we can remove it from negatives
                try:
                    train_negative_img.remove(int(arr[0][:5]))
                except ValueError:
                    pass
                parse_sample_information(arr, train_dict)
            elif int(arr[0][:5]) < 600:
                try:
                    valid_negative_img.remove(int(arr[0][:5]))
                except ValueError:
                    pass
                parse_sample_information(arr, valid_dict)
            else:
                try:
                    test_negative_img.remove(int(arr[0][:5]))
                except ValueError:
                    pass
                parse_sample_information(arr, test_dict)

    train_negatives = {
        'names': [],
    }
    valid_negatives = {
        'names': [],
    }
    test_negatives = {
        'names': [],
    }
    # create image name from known index number, for training set
    for image_number in train_negative_img:
        train_negatives['names'].append(find_full_sample_name(image_number))
    for image_number in valid_negative_img:
        valid_negatives['names'].append(find_full_sample_name(image_number))
    for image_number in test_negative_img:
        test_negatives['names'].append(find_full_sample_name(image_number))


    # return 2 touples, one contain information about test set and other about train set
    return (train_dict, train_negatives),(valid_dict, valid_negatives) , (test_dict, test_negatives)


def germ_super_class(class_num):

    super_class = {
        # sign classes in shape of circle
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        # sign classes in shape of triangle
        1: [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        2: [12],  # sign class shaped as diamond
        3: [13],  # sign class in shape of verticali flipped triangle
        4: [14],  # sign class shaped as octagon
    }
    for super_c, classes in super_class.items():
        if class_num in classes:
            return super_c
    # you should not be here
    raise RuntimeError("Super class for class{} not found.".format(class_num))


def find_full_sample_name(image_number):
    # create name from number
    # this apply for GMSDB dataset
    # examples. 10 -> 00010.ppm, 105 -> 00105.pm

    image_number = str(image_number)
    zeros = ''.join(['0' for x in range(5 - len(image_number))])
    whole_name = zeros + image_number + ".ppm"
    return whole_name

#########################End of GTSD functions##################################


def coordinates_to_bb_data(coor):
    """
    TODO I dont know if I should add +1 here
    Get coordinates as -> [x1,y1,x2,y2], compute bounding box data as -> [x1, y1, h, w]
    CANT BE USED WITH NORMALIZED VALUES 
    """
    x1 = coor[0]
    y1 = coor[1]
    w = coor[2] - coor[0] + 1
    h = coor[3] - coor[1] + 1
    
    return [x1,y1, h, w]

def bb_data_to_coordinates(data):
    """
    Get bb data as -> [x1,y1,h,w], compute coordinates as -> [x1, y1, x2, y2]
    """
    x1 = data[0]
    y1 = data[1]
    x2 = data[0] + data[3] - 1
    y2 = data[1] + data[2] - 1

    return [x1,y1, x2, y2]




def parse_sample_information(sample_params, samples_dict):
    # from sample_params parse basic informations about sample
    # such ass name, coordinates and save it to samples_dict

    image_name = sample_params[0]

    if image_name not in samples_dict.keys():
        samples_dict[image_name] = []

    
    #!!!! in case you will ever change this, take care of creating dict from this data
    #in function adjust_sign_coor in func.py model
    samples_dict[image_name].append(
        {'class': sample_params[5],
         'super-class': sample_params[6],
         'coordinates' : sample_params[1:5]
        }
    )

def generate_dataset(dataset_type, dataset_path, new_dataset_path, image_size):

    if dataset_type == "class":
        data_proc = DatasetGenerator(new_dataset_path)
        train_dataset, test_dataset = class_dataset_wrapper(dataset_path)
        create_randomly(train_dataset,os.path.join(dataset_path, "Final_Training", "Images"), data_proc, mode="train", size=image_size)
        #create_randomly(test_dataset,os.path.join(dataset_path, "Final_Test", "Images"), data_proc, mode="test", size=image_size)

    elif dataset_type == "german":
        train_dataset, valid_dataset, test_dataset = german_dataset_wrapper(dataset_path)
        data_proc = DatasetGenerator(new_dataset_path)
        # creating training data
        #create_positives(train_dataset[0],dataset_path, data_proc, mode="train", size=image_size, full=1, crop=1, w_background=0)
        create_negatives(train_dataset[1], dataset_path, data_proc, mode="train", size=image_size, goal_amount=60000)
        # creating test_data
        #create_positives(test_dataset[0],dataset_path, data_proc, mode="test", size=image_size, full=1, crop=1, w_background=0)
        create_negatives(valid_dataset[1], dataset_path, data_proc, mode="test", size=image_size, goal_amount=10000)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data', choices=['german', 'belgium', "class"], help='type of dataset')
    parser.add_argument('load', help='path to old dataset')
    parser.add_argument('new', help='path to new dataset')
    parser.add_argument('size', type=int, help='size of samles')
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.load)
    new_dataset_path = os.path.abspath(args.new)
    generate_dataset(args.data, dataset_path, new_dataset_path, args.size)
