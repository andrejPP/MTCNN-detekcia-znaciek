import os
from mtcnn import MTCNN
from image import draw_boxes, show_image, image_load

def detection(image_name):
    """
    Run detector on single image
    """
    detector = MTCNN()

    try:
        return  detector.detect(image_name)
    except ValueError:
        print("No sign detected")
        return []

def run_demo():
    """
    Run demo on images in folder "images"
    """

    path = os.path.abspath("../images")
    image_list = os.listdir(path)
    

    for image_name in image_list:
        image_path = os.path.join(path, image_name)
        print("-----------------------------------------------")
        print("Path:", image_path)
        image = image_load(image_path)
        detect_data = detection(image_path)

        if len(detect_data) > 0:
            draw_boxes(image, detect_data[:,0:4], detect_data[:,4], detect_data[:,5], red=255)
        show_image(image, False)
        print(detect_data)

if "__main__" == __name__:

    run_demo()
