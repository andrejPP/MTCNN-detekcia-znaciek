########################################################
#      
#      Autor: Andrej Paníček           
#      Posledná zmena: 10.5.2019
#      Popis: Tento modul obsahuje triedu Image, tá načíta
#      a škáluje obrázky pre detektor MTCNN.
#
########################################################


import cv2
import sys
import numpy as np

def image_load(image_path, channel_order="RGB"):

    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise Exception("Couldn't load image: " + image_path)
    #opencv default order is BGR
    return change_channel_order(image, current="BGR", new=channel_order)

def change_channel_order(image, current, new):

    available_orders = ["RGB", "BGR"]

    #change order of channels from BRG to RGB
    if (new not in available_orders) or (current not in available_orders):
        raise ValueError("Unsuported order of colors")
    #cv2  defaul order is BGR
    if new == current:
        return image
    elif new == "RGB":
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    elif new == "BGR":
        return cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


def unnormalize_image(image):
    #reverse of normalize image function
    image = image / 0.0078125 + 127.5
    return np.uint8(image)


def normalize_image(image):

    image = (image -127.5) * 0.0078125
    return image

def show_image(image,normalized=True):

    if normalized:
        image = unnormalize_image(image)
    if Image_processor.channel_order == "RGB":
        image = change_channel_order(image, current="RGB", new="BGR")
    
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_boxes(image, boxes, score=None, class_num=None, blue=0, green=0, red=0, thickness=2):
    """
    get bounding boxes and show them
    """
    if score is not None:
        score = np.round(score, 2)

    show_image = image
    
    print("Number of boxes:", len(boxes))
    for index, (start_x, start_y, end_x, end_y) in enumerate(boxes, 0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if score is not None:
            cv2.rectangle(show_image, (int(float(start_x)), int(float(start_y)-20)), (int(float(start_x)+45), int(float(start_y))), (blue, green, red), -1)
            cv2.putText(show_image,"CL: " + str(int(class_num[index])),(int(start_x),int(start_y)-4), font, 0.4,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(show_image, (int(float(start_x)), int(float(start_y))), (int(float(end_x)), int(float(end_y))), (blue, green, red), thickness)

    return show_image


class Image_processor:

    channel_order = "RGB"
    channels = 3

    def __init__(self, image_path, signs_coor=None):
        """

        Args:
        signs_coor - array containing coordinates of sings, inside image
            it is used with training, when we need to know, where are the signs appear
            example for single sign [ { 'class': '27', 'coordinates': [x1,y1,h,w]} ]
        """
        #single precision, so we dont need to fight with net (.to(torch.float64))
        #also this way it's eating less memory
        # self._image = np.float32(image)
        self._orig_image = image_load(image_path, Image_processor.channel_order)
        self._image = normalize_image(self._orig_image)
        # print(self._image)
        # print(type(self._image))
        self._min_detect_size = None
        self._min_obj_size = None
        self._img_scales = list()
        self._signs_coor = signs_coor

    def full_image(self):
        #TODO this function is used when we create negatives from false positives
        #in process_dataset.py
        return self._orig_image

    def height(self):
        return self._image.shape[0]

    def width(self):
        return self._image.shape[1]

    def crop_picture(self, x1, y1, x2,y2):
        """
        indicies represents, x1, y1, x2, y2
        """

        if (x1 < 0) or (y1 < 0):
            raise Exception("Image.py crop_picture method, out of bounderies.")
        elif (x2 >= self.width()) or (y2 >= self.height()):
            raise Exception("Image.py crop_picture method, out of bounderies.")

        crop_image = self._image[int(y1):int(y2)+1, int(x1):int(x2)+1]
        return crop_image


    def create_pyramide(self, scale_arguments):
        # self._min_detect_size =  detectable_size
        # self._min_obj_size =  min_obj_size
        self._scale_image(**scale_arguments)
        output = []

        for scale in self._img_scales:
            new_height = int(np.ceil(scale * self._image.shape[0]))
            new_width = int(np.ceil(scale * self._image.shape[1]))
            scaled_image = np.float32(cv2.resize(self._image, dsize=(new_width, new_height)))

            #normalize image , range
            # scaled_image = (scaled_image - 127.5) / 127.5
            output.append({'scale': scale, 'image':scaled_image })

            # cv2.imshow('image',scaled_image)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

        return output

    def _scale_image(self, **kwargs):
        '''
        return valid scales of picture
        '''

        factor = kwargs['factor']
        min_detect_size = kwargs['min_detect_size']
        min_object_size = kwargs['min_object_size']

        min_img_length = min(self._image.shape[0], self._image.shape[1])

        #we need to resize image,
        #by scale of "min detection size"/"min object size we want"
        #for example : min_detect_size == 12, min_object_size = 20
        #   12/20 = 0.6
        # so we compute scales from  0.6 size of original image,
        # when we detect object as small as 12x12  by multiplaying by 0.6
        # we get our smallest possible size of object we wanna detect
        base_size = min_detect_size/min_object_size
        min_img_length *= base_size

        #scale image
        scale_index = 0
        while min_img_length > min_detect_size:
            self._img_scales.append(base_size*(factor**scale_index))
            min_img_length *= factor
            scale_index += 1

    def signs_coor(self):
        return self._signs_coor


    def __str__(self):

        rounded_scales = [round(scale, 3) for scale in self._img_scales]

        string = "Image shape: " + str(self._image.shape)  + "\n"
        string += "Image scales: " + str(rounded_scales)
        return string
