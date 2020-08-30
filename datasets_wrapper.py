import os

def classification_dataset_wrapper(dataset_path):
    """
    Function for GTSRB (classification) dataset that reads 
    samples information from ground truth info file. 

    Args:
        dataset_path: path to GTSRB dataset root dir
    Returns:
        Two dictionary, first one contains information about train set
        and the second one contains information about set set.
    """

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


def detection_dataset_wrapper(dataset_path):
    """
    Function for GTSDB (detection) dataset that reads samples information 
    from ground truth info file. 

    Args:
        dataset_path: path to GTSDB dataset root dir
    Returns:
        Three touples, first one contains information about train set,
        the second one validataion set and the final one about test set.
    """

    info_file = os.path.join(dataset_path, "gt.txt")
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
                # Any image that is in info file, does contain at least 1 sign,
                # therefore we can remove it from negatives.
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
    # Create image name from known index number, for training set
    for image_number in train_negative_img:
        train_negatives['names'].append(find_full_sample_name(image_number))
    for image_number in valid_negative_img:
        valid_negatives['names'].append(find_full_sample_name(image_number))
    for image_number in test_negative_img:
        test_negatives['names'].append(find_full_sample_name(image_number))

    return (train_dict, train_negatives), (valid_dict, valid_negatives), (test_dict, test_negatives)


def germ_super_class(class_num: int):
    """
    For GTSDB dataset get super class of class specified by parameter class_num.
    Super classes are different shapes of signs.

    Args:
        class_num: class index
    Return:
        Index of super class.
    """

    super_class = {
        # Classes with shape of circle.
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        # Classes with shape of triangle.
        1: [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        2: [12],  # Class shaped as diamond.
        3: [13],  # Class shaped as verticali flipped triangle.
        4: [14],  # Class shaped as octagon.
    }
    for super_c, classes in super_class.items():
        if class_num in classes:
            return super_c

    raise RuntimeError("Super class for class{} not found.".format(class_num))


def find_full_sample_name(image_number):
    """
    Create name for image file with number specified by image_number parameter.
    e.g. image_number=10 -> 00010.ppm, image_number=105 -> 00105.pm

    Args:
        image_number: sample number that should be included in name of the file
    Return:
        Full name for image sample with .ppm extension.
    """

    image_number = str(image_number)
    zeros = ''.join(['0' for x in range(5 - len(image_number))])
    whole_name = zeros + image_number + ".ppm"
    return whole_name


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