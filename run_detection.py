import numpy as np
import sys
import os
from mtcnn_new import MTCNN
from image import draw_boxes, show_image, image_load

detector = MTCNN()

def detection(image_name):
    """
    Run detector on single image.
    """

    try:
        return  detector.detect(image_name)
    except ValueError:
        print("No sign detected")
        return []


def run_demo():
    """
    Run demo on images in folder "images"
    """

    path = os.path.abspath("./images")
    image_list = os.listdir(path)
    
    print(f"----Runnng demo on {len(image_list)} images from GTSDB dataset.----")
    for image_name in image_list:
        image_path = os.path.join(path, image_name)
        print("-----------------------------")
        print("Path:", image_path)
        run_single_detection(image_path)
    

def run_single_detection(image_path):
    """
    Load image, run detection and draw detected bounding boxes.
    """
    
    image_path = os.path.abspath(image_path)
    image = image_load(image_path)
    bbox_data = detection(image_path)
    show_result(bbox_data, image)


def show_result(bbox_data, image):

    if len(bbox_data) > 0:
        draw_boxes(image, bbox_data[:, 0:4], bbox_data[:, 4], bbox_data[:, 5], red=255)
    show_image(image, False)
    print(bbox_data)


if "__main__" == __name__:

    if len(sys.argv) == 1:
        # No image path as argumet, so run demo.
        run_demo()
    else:
        # Run detection on arguments (each argument should be path to image).
        for image_path in sys.argv[1:]:
            run_single_detection(image_path)

