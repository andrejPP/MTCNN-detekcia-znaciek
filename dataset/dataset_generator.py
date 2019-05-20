########################################################
#      
#      Autor: Andrej Paníček           
#      Posledná zmena: 10.4.2019
#      Popis: Tento modul obsahuje triedu 
#      DatasetGenerator, tá ukladá dátovú sadu. 
#                              
#
########################################################

import os
import csv 
import cv2

class DatasetGenerator():

    def __init__(self, dataset_path):
        self._csv_filename = "info.csv"
        self._dataset_path = None
        self._modes = ["train","test"]
        self._sample_types = ["positives", "negatives", "parts"]
        self._initialize_folders(dataset_path)

    def _initialize_folders(self, dataset_path):
        """
        Create folder structure for dataset.
        args:
            dataset_path - the root directory of dataset, should be also created 
        """
        self._dataset_path = self._create_folder(dataset_path)
        # create folders for training data and testing data
        for sample_type in self._sample_types:
            self._create_folder(os.path.join(
                self._dataset_path, self._modes[0], sample_type))
            self._create_folder(os.path.join(
                self._dataset_path, self._modes[1], sample_type))

    def _create_folder(self, folder_name):
        """
        Create single folder.
        args:
            folder_name - folder path, which should be created
        """
        folder_name = os.path.abspath(folder_name)
        try:
            # create folder and also csv file, for information about images
            os.makedirs(folder_name)
            os.mknod(os.path.join(folder_name, self._csv_filename))

            print(">> {} folder successfuly created.".format(folder_name))
        except FileExistsError:
            pass
            # print(">> {} folder already exist.".format(folder_name))
        return folder_name

    def _save_info_csv(self, path, information):
        """
        Save information about images into file.

        Args:
            path: directory where we find csv file and save information
            information: tuple(image name, list of coordinates)
        """

        path = os.path.join(path, self._csv_filename)
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(information)

    def _save(self, path, name, image, info_tuple):
        """
        Save image in directory defined by path. And also update .csv file.

        Args:
            path - directory where we save our file
            image - image we'll save
            info_tuple - information about image
        """

        pic_path = os.path.join(path, name)
        cv2.imwrite(pic_path, image)
        print("Saved picture: ", pic_path)
        self._save_info_csv(path, info_tuple)

    def save_img(self, image, sample_type, coordinates, box_height, box_width, mode, base_class=None, super_class=None):
        """
        There we find out where and how to save image in our dataset and then we
        save it properly.

        args:
            image - image we want to save
            sample_type - negatives/positives/parts
            base_class  - sign class
            super_class - super class of sign
            coordinates - coordiantions of bounding box - x1, y1, x2, y2
            box_height - height of ground truth bounding box
            box_width - width of ground truth bounding box

        """
        # based on mode set folders
        if mode not in self._modes:
            raise ValueError("Wrong mode: {}".format(mode))
        if sample_type not in self._sample_types:
            raise ValueError("Unknown sample type: {}".format(sample_type))
        # type "positives" has to be in combination with parameters base_class and super_class
        if sample_type == "positives":
            # both params have to be set
            if (base_class is None) or (super_class is None):
                raise ValueError("Expected parameters base_class and super_class to be set.")

        # create path for saving this image, based on parameter mode
        # and parameter sample_type
        #e.g. sample_type = negatives, mode=train
        #   path'll be ->  dataset_base_path/train/negatives
        path = os.path.join(self._dataset_path, mode, sample_type)

        x1, y1, x2, y2 = coordinates
        # save only as negative sample
        pic_name = self._gen_pic_name(path)
        information_tuple = (pic_name, x1, y1, x2, y2, box_height, box_width, super_class, base_class)
        self._save(path=path,
                   name=pic_name,
                   image=image,
                   info_tuple=information_tuple)

    def _gen_pic_name(self, class_path):
        """
        List all files and generate new name
        """
        number_of_pic = str(len(os.listdir(class_path)))
        return number_of_pic + ".png"

