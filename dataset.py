
import os
import sys
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from process_dataset import normalize_coordinates


class SignDataset(Dataset):
    """ 
    Class representing sign dataset structure I used for training MTCNN.

    Args:
        dataset_dir: path to the directory root containing datased
        net: what type of neural network is trianed
        classification: one of the three possible classification mode
        mode: either "train" or "test"
        transform: transform function, which is applied to each sample
    """
    # The key is type of the network and value is size of sample dimensions.
    nets = {"first": 12,
            "second": 24,
            "third": 48 }
    # Possible classification modes. 
    # binary: two classes, either sign or background
    # super: signs are separated into classes based on the shape
    # multi: each sign is separated class
    classification_modes = ["binary", "super", "multi"]

    def __init__(self, dataset_dir, net, classification, mode="train", transform=None):
        super().__init__()
        self._class_mode = classification
        self._mode = mode
        self._img_size = SignDataset.nets[net]
        self._net = net
        self._transform = transform
        self._dataset_dir = os.path.join(os.path.abspath(dataset_dir), mode)
        self._samples, self._class_samples_idx = self._load_classes()

        if net not in SignDataset.nets:
            raise RuntimeError("Not available for net called:", net)
        if classification not in SignDataset.classification_modes:
            raise RuntimeError(
                "Unknown classification mode ->", classification)

    def _load_classes(self):
        """
        Load samples and separate them into classes. This 
        time class can be one of parts, positives, negatives
        (not talking about sign classes). 

        Return:
            List of all samples and directory storing index
            of sample where each class starts. 
        """
        class_sample_idx = {}
        samples = []
        classes = {-1: "parts",
                   0: "negatives",
                   1: "positives"}

        for class_index in sorted(classes.keys()):
            class_name = classes[class_index]
            class_sample_idx[class_name] = [len(samples)]

            # Load samples from class directory,
            class_path = os.path.join(self._dataset_dir, class_name)
            new_samples = self._load_samples(class_path, class_index)
            samples.extend(new_samples)
            class_sample_idx[class_name].append(len(samples)-1)

        return samples, class_sample_idx

    def _load_samples(self, class_path, class_number):
        """
        Load samples for class (either part, positive or negative).
        For each sample we save tuple (image path, class index, x1,y1,x2,y2 ).

        Args:
            class_path: path to root directory of current class
            class_number: current class number, class number mapping
                can be seen in _load_classes method
        Return:
            List of loaded samples.
        """

        samples = []
        info_file = os.path.join(class_path, "info.csv")

        with open(info_file, newline='') as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                label = class_number
                # Find image path.
                image_path = os.path.join(class_path, row[0])
                # Get x1 and y1 coordinates,height and weight.
                coor = np.array(row[1:5], dtype=np.float32)
                coor = normalize_coordinates(norm_max=1, width=self._img_size, height=self._img_size, coor=coor)
                coor = torch.FloatTensor(coor)

                # When loading positive samples, we also need to 
                # setup label (sign class) properly.
                if class_number == 1:
                    if self._class_mode == "binary":
                        pass
                    elif self._class_mode == "super":
                        label += int(row[7])
                    elif self._class_mode == "multi":
                        label += int(row[8])

                samples.append([label, image_path, coor])
        return samples

    def dataset_info(self):
        """
        Get inforormation where dataset classes (part, negative, positive) 
        samples start in list of all samples.

        Return:
        Dictionary with class number as keys and value is index number in
        self._samples attribute.
        """

        return self._class_samples_idx

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        """
        Get data from specific index position in dataset.
        """
        label, path, coor = self._samples[index]
        image = pil_loader(path)
        if self._transform is not None:
            image = self._transform(image)
        return image, label, coor


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        #print(list(img.getdata()))
        return img.convert('RGB')

def normalize_regresion_input(x, y, h, w, image_size):
    """
    Args:
        x: horizzontal starting position of bounding box
        y: vertical starting position of bounding box
        h: height of bounding box
        w: width of bounding box
        image_size: size of input image 
    """
    h -= 1
    w -= 1
    ratio = 1/(image_size-1)

    norm_x = x * ratio
    norm_y = y * ratio
    norm_h = h * ratio
    norm_w = w * ratio

    return np.array([norm_x,norm_y,norm_h,norm_w], dtype=np.float32)

# if __name__ == "__main__":
#    path = "./datasets/new_format"
#    dataset = Sign_dataset(path,"first", classification="binary")
#    print(dataset.dataset_info())
#    dataset = Sign_dataset(path,"first", classification="super")
#    print(dataset.dataset_info())
#    dataset = Sign_dataset(path,"first", classification="multi")
#    print(dataset.dataset_info())
