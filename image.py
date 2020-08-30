import cv2
import sys
import numpy as np

#TODO CHANGE NAME OF THE CLASS TO PROPER FORMAT

def image_load(image_path, channel_order="RGB"):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise Exception("Couldn't load image: " + image_path)
    #opencv default order is BGR
    return change_channel_order(image, current="BGR", new=channel_order)

def change_channel_order(image, current, new):
    available_orders = ["RGB", "BGR"]
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
    # Normalize pixels value to range (255,0), round values to closest integer.
    image = image / 0.0078125 + 127.5
    return np.uint8(image)


def normalize_image(image):
    # Normalize pixels value to range (-1,1). Creates float representation.
    image = (image -127.5) * 0.0078125
    return image

def show_image(image,normalized=True):
    if normalized:
        image = unnormalize_image(image)
    if ImageProcessor.channel_order == "RGB":
        image = change_channel_order(image, current="RGB", new="BGR")
    
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_boxes(image, boxes, score=None, class_num=None, red=0, green=0, blue=0, thickness=2):
    """
    Draw bounding boxes with class name and class probability. 
    """
    if score is not None:
        score = np.round(score, 2)

    show_image = image
    
    print("Number of boxes:", len(boxes))
    for index, (start_x, start_y, end_x, end_y) in enumerate(boxes, 0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if score is not None:
            cv2.rectangle(show_image, (int(float(start_x)), int(float(start_y)-20)), (int(float(end_x)), int(float(start_y))), (blue, green, red), -1)
            cv2.putText(show_image, str(int(class_num[index])),(int(start_x),int(start_y)-4), font, 0.4,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(show_image, (int(float(start_x)), int(float(start_y))), (int(float(end_x)), int(float(end_y))), (blue, green, red), thickness)

    return show_image


class ImageProcessor:
    """
    Class representing an image with all needed operations on it.
    Args:
        image_path: path to the image to be loaded
        signs_coor: list containing coordinates of sings in this image,
            it is used while training, when we need to know positions and classes of
            signs inside this image
            (e.g.) for single sign [ { 'class': '27', 'coordinates': [x1,y1,h,w]} ]
    """

    channel_order = "RGB"
    channels = 3

    def __init__(self, image_path, signs_coor=None):
        self._orig_image = image_load(image_path, ImageProcessor.channel_order)
        self._image = normalize_image(self._orig_image)
        self._min_detect_size = None
        self._min_obj_size = None
        self._img_scales = list()
        self._signs_coor = signs_coor

    def full_image(self):
        """
        Return:
            Original image.
        """

        return self._orig_image

    def height(self):
        return self._image.shape[0]

    def width(self):
        return self._image.shape[1]

    def crop_picture(self, x1, y1, x2, y2):
        """
        Crop region from the image, specified by parameters.
        """

        if (x1 < 0) or (y1 < 0):
            raise Exception("Image.py crop_picture method, out of bounderies.")
        elif (x2 >= self.width()) or (y2 >= self.height()):
            raise Exception("Image.py crop_picture method, out of bounderies.")

        crop_image = self._image[int(y1):int(y2)+1, int(x1):int(x2)+1]
        return crop_image

    def create_pyramide(self, scale_arguments):
        self._scale_image(**scale_arguments)
        output = []

        for scale in self._img_scales:
            new_height = int(np.ceil(scale * self._image.shape[0]))
            new_width = int(np.ceil(scale * self._image.shape[1]))
            scaled_image = np.float32(cv2.resize(self._image, dsize=(new_width, new_height)))

            output.append({'scale': scale, 'image':scaled_image })
        return output

    def _scale_image(self, factor, min_detect_size, min_object_size):
        '''
        Calculate proper scales for image pyramide.
        Args:
            factor: 
            min_detect_size: minimal detectable size (in pixels) of sign in image
            min_object_size: minimal size (in pixels) of sign we are interested in

        Let's say we want to detect only signs bigger than 20x20 pixels and
        mininaml size detectable by network is 12x12 pixels. By dividing 
        12/20 we get 0.6, so we will take 0.6 times original size of the image
        as a base value in calculation of scales.
        '''

        min_img_length = min(self._image.shape[0], self._image.shape[1])
        base_size = min_detect_size/min_object_size
        min_img_length *= base_size

        scale_index = 0
        while min_img_length > min_detect_size:
            self._img_scales.append(base_size*(factor**scale_index))
            min_img_length *= factor
            scale_index += 1

    def signs_coor(self):
        """
        Return:
            coordinates of signs in image     
        """

        return self._signs_coor

    def __str__(self):
        rounded_scales = [round(scale, 3) for scale in self._img_scales]
        string = "Image shape: " + str(self._image.shape)  + "\n"
        string += "Image scales: " + str(rounded_scales)
        return string
