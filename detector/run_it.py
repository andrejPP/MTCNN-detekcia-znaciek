########################################################
#      
#      Autor: Andrej Paníček           
#      Posledná zmena: 11.5.2019
#      Popis: Tento modul obsahuje pomocné funkcie, a testovacie 
#      pre tridu MTCNN.
#                              
#
########################################################

import sys
import os

import numpy as np
from mtcnn import MTCNN
from process_dataset import german_dataset_wrapper


def run_single_detection(image_name):
    """
    Run detector on one single image
    """
    #Create main detector object, which control whole process
    detector = MTCNN(gpu=True)

    image_name = '/home/andrej/bakalaris/mtcnn/my_mtcnn/images/twitter.jpg'
    detector.detect(image_name)

def run_on_dataset(dataset_path, dataset_info):

    detector = MTCNN(gpu=True)

    images_with_sign = dataset_info[0]
    images_without_sign = dataset_info[1]
#    print(images_without_sign)
#    for image_name, data in images_without_sign.items():
#        for each in  data:
#            image_name = dataset_path + each
#
#            try:
#                detector.detect(image_name)
#            except ValueError:
#                print("No boxes detected.")
#                continue


    print("Dataset path:",dataset_path)
    for image_name, signs_info in images_with_sign.items():

        image_name = os.path.join(dataset_path, image_name)
        try:
            detector.detect(image_name, signs_info)
        except ValueError:
            print("No boxes detected.")
            
            continue


def create_dataset(dataset_path, new_dataset_path, layer, training_info, testing_info):

    detector = MTCNN(gpu=True)
    modes = {"train" : training_info,
            "test" : testing_info}

    #create training and testing part of datset
    for mode, info in modes.items():
        if info is None:
            continue
        images_with_sign = info[0]
        images_without_sign = info[1]

        print("Dataset path:",dataset_path)
        for image_name, signs_info in images_with_sign.items():


            image_name = os.path.join(dataset_path,image_name)
            detector.extract_output_samples(image_name, 
                                            signs_info, 
                                            layer,
                                            new_dataset_path,
                                            mode)


def get_info(info_file, gt_path, dt_path, dataset_path):


    detector = MTCNN(gpu=True)
    with_sign = info_file[0]
    without_sign = info_file[1]
#    for image_name in without_sign['names']:
#        image = image_name.split(".")[0]
#        file_name = os.path.join(gt_path, image+".txt")
#        with open(file_name, "w") as f:
#            f.write("\n")
#            pass
#
#        image_name = os.path.join(dataset_path, image_name)
#        try:
#           result = detector.detect(image_name)
#        except ValueError:
#            file_name = os.path.join(dt_path, image+".txt")
#            with open(file_name, "w") as f:
#                f.write("\n")
#            continue
#
#
#        file_name = os.path.join(dt_path, image+".txt")
#        print(file_name)
#
#        with open(file_name, "w") as f:
#
#            for each in result:
#                score = str(each[4])
#                class_name = str(int(each[5]))
#                x1 = str(int(each[0]))
#                y1 = str(int(each[1]))
#                x2 = str(int(each[2]))
#                y2 = str(int(each[3]))
#
#                f.write(class_name + " " + score  +" "+ x1 +" " + y1 +" " + x2 +" " + y2 + "\n")


    #now the same with pictures of sign
    for image_name, signs_info in with_sign.items():

        #first ground truth boxes
        image = image_name.split(".")[0]
        file_name = os.path.join(gt_path, image+".txt")

        with open(file_name, "w") as f:
            for sign in signs_info:
                class_name = sign['class']
                coordinates = sign['coordinates']
                x1 = coordinates[0]
                y1 = coordinates[1]
                x2 = coordinates[2]
                y2 = coordinates[3]
                f.write(class_name +" "+ x1 +" " + y1 +" " + x2 +" " + y2 + "\n")


        #secondly detection boxes
        image_name = os.path.join(dataset_path, image_name)
        try:
           result = detector.detect(image_name, signs_info)
        except ValueError:
            file_name = os.path.join(dt_path, image+".txt")
            with open(file_name, "w") as f:
                f.write("\n")
                pass
            continue


        file_name = os.path.join(dt_path, image+".txt")
        print(file_name)

        with open(file_name, "w") as f:

            for each in result:
                score = str(each[4])
                class_name = str(int(each[5]))
                x1 = str(int(each[0]))
                y1 = str(int(each[1]))
                x2 = str(int(each[2]))
                y2 = str(int(each[3]))

                f.write(class_name + " " + score  +" "+ x1 +" " + y1 +" " + x2 +" " + y2 + "\n")



if __name__ == '__main__':
    pass
    
    #Can run GTSBD, dataset_path has to be path to GTSBD dataset
    #dataset_path = '/home/andrej/bakalaris/mtcnn/my_mtcnn/FullIJCNN2013/'
    #training_info, validation_info, testing_info = german_dataset_wrapper(dataset_path)
    #run_single_detection("path to image, have to be here")
    #create_dataset(dataset_path, "./datasets/last_chance", 0, training_info,  validation_info)
    #run_on_dataset(dataset_path, validation_info)
    #run_on_dataset(dataset_path, testing_info)
 
