import os
import csv 
import cv2

class DatasetGenerator():
    """
    Class representing dataset. Create or process dataset structure and 
    correctly save images and ground truth information. 

    Args:
        dataset_path: path to dataset
    """

    def __init__(self, dataset_path):
        self._csv_filename = "info.csv"
        self._dataset_path = None
        self._modes = ["train","test"]
        self._sample_types = ["positives", "negatives", "parts"]
        self._initialize_folders(dataset_path)

    def _initialize_folders(self, dataset_path):
        """
        Create folder structure for dataset.

        Args:
            dataset_path: the root directory of dataset, should be also created 
        """
        self._dataset_path = self._create_folder(dataset_path)
        # Create folders for training data and testing data.
        for sample_type in self._sample_types:
            self._create_folder(os.path.join(
                self._dataset_path, self._modes[0], sample_type))
            self._create_folder(os.path.join(
                self._dataset_path, self._modes[1], sample_type))

    def _create_folder(self, folder_name):
        """
        Create single folder.

        Args:
            folder_name - folder path, that should be created
        """
        folder_name = os.path.abspath(folder_name)
        try:
            # Create folder csv file for ground truth information.
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
            path: directory where we save our file
            image: image to save
            info_tuple" information about image
        """

        pic_path = os.path.join(path, name)
        cv2.imwrite(pic_path, image)
        print("Saved picture: ", pic_path)
        self._save_info_csv(path, info_tuple)

    def save_img(self, image, sample_type, coordinates, box_height, box_width, mode, base_class=None, super_class=None):
        """
        Validate all required information about image and correctly save 
        them along the image to the dataset.  

        Args:
            image: image we want to save
            sample_type: type of image, one of (negatives/positives/parts)
            base_class: sign class
            super_class: super class of sign (super classes 
                are divided by shape of sign)
            coordinates: coordinates of bounding box - x1, y1, x2, y2
            box_height: height of ground truth bounding box
            box_width: width of ground truth bounding box
        """

        if mode not in self._modes:
            raise ValueError("Wrong mode: {}".format(mode))
        if sample_type not in self._sample_types:
            raise ValueError("Unknown sample type: {}".format(sample_type))
        # Type "positives" has to be in combination with parameters base_class and super_class
        if sample_type == "positives":
            if (base_class is None) or (super_class is None):
                raise ValueError("Expected parameters base_class and super_class to be set.")

        # Create path where this image will be saved, based on parameter mode
        # and parameter sample_type.
        # e.g. sample_type = negatives, mode = train
        #   created path will be  ->  dataset_base_path/train/negatives
        path = os.path.join(self._dataset_path, mode, sample_type)

        x1, y1, x2, y2 = coordinates
        pic_name = self._gen_pic_name(path)
        information_tuple = (pic_name, x1, y1, x2, y2, box_height, box_width, super_class, base_class)
        self._save(path=path,
                   name=pic_name,
                   image=image,
                   info_tuple=information_tuple)

    def _gen_pic_name(self, class_path):
        """
        Generate unique image name, based on current index in dataset.
        """
        number_of_pic = str(len(os.listdir(class_path)))
        return number_of_pic + ".png"

