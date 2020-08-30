
from mtcnn_new import MTCNN
import os
from datasets_wrapper import  detection_dataset_wrapper
from func import load_model, save_json, find_matches
import numpy as np
import argparse
from nets import FirstNet, SecondNet, LastNet
from experiments.ploting_exp import plot_roc


def init_mtcnn(model, net, class_mode, threshold=None):
    """
    Instantiate mtcnn and call proper setup function.  
    """

    detector = MTCNN(gpu=True)
    setup_nets(detector, model, class_mode, net, threshold)
    return detector


def setup_nets(detector, model, class_mode, net, threshold, channels=3):
    """
    Properly setup Mtcnn instance for evaluation. Instead of default model
    for one of the nets, use model received as parameter.  

    Args:
        detector: MTCNN instance
        model: model to replace default one
        class_mode: one from (mutli, binary, super)
        net: string, which net model change for the one received in parameter
        threshold: lower IOU boundary for positive matches
        channels: how many channels input image has
    """

    if net == "first":
        net = FirstNet(class_mode, channels=channels)
        load_model(net, model, mode="test")
        detector.first_net = net
        if threshold is not None:
            detector._thresholds[0] = threshold
    elif net == "second":
        net = SecondNet(class_mode, channels=channels)
        load_model(net, model, mode="test")
        detector.sec_net = net
        if threshold is not None:
            detector._thresholds[1] = threshold
    elif net == "third":
        net = LastNet(class_mode, channels=channels)
        load_model(net, model, mode="test")
        detector.last_net = net
        if threshold is not None:
            detector._thresholds[2] = threshold
    else:
        raise ValueError("Not allowed value for parameter \"level\" -> ",net)


def write_output(information):

    print("Summary:")
    print("==========================")
    print(f"Model used {information['model']}.")
    print(f"Number of boxes generated: {information['box_generated']}.")
    print(f"Number of positive boxes: {information['positive'][0]}, which is {information['positive'][1]} %.")
    print(f"Number of false positive boxes: {information['false positive'][0]}, which is { information['false positive'][1]} %.")
    print(f"Number of part sign boxes: {information['parts'][0]}, which is  {information['parts'][1]} %.")
    print(f"{information['images']} images checked.")
    print(f"Total: {information['total_sign']}.")
    print(f"Correct: {information['detected']}.")
    print(f"Accuracy on test dataset with IOU {information['iou_positive']} is {100 * information['detected'] / information['total_sign']:.4f} %. ")
    print("==========================")



def detected(sign_coordinates, boxs_coordinates, threshold=0.65):
    """ 
    Extract bounding boxes that match GT into two groups
    1. positive (valid match, threshold set by parameter to 1.0)
    2. parts (partly match, 0.40 to positive threshold)

    Args:
        sign_coordinates: coordinates of sign ground truth box
        boxs_coordinates: coordinates of detected bounding boxes
        threshold: lower boundary for IOU of positive bounding boxes
    Return:
        Two lists, the first one contain indicies of bounding boxes classified as positive matches.
        The Second one contains indicies of bounding boxes classified as partly match.
    """
    pos = []
    parts = []
    
    part_iou = 0.40
    if threshold < part_iou:
        raise RuntimeError("IOU threshold for positive detections can't be lower than 0.4.")
    elif threshold == part_iou:
        print("IOU thresholds for positive and part are set to same value.")

    boxes_sign_iou = find_matches(boxs_coordinates, sign_coordinates, iou_threshold=part_iou)
    for box_index, data in boxes_sign_iou.items():
        if data['iou'] >= threshold:
            pos.append(box_index)
        elif data['iou'] >= part_iou:
            parts.append(box_index)
    return pos, parts

#  TODO waiting for refactorization
#def roc_benchmark(model, net, threshold, dataset_info, dataset_path, class_mode#):
#
#    detector = init_mtcnn(model, net, class_mode, threshold=1)
#    images_with_sign = dataset_info
#
#    layers =  {
#        "first" : 0,
#        "second" : 1,
#        "third" : 2
#    }
#
#    roc_database = {}
#    score_tresh = [1,0.95, 0.9,0.85, 0.8,0.75, 0.7,0.65, 0.6,0.55, 0.5,0.45, 0.4,0.35, 0.3, 0.2, 0.15, 0.1, 0.05,0]
#
#    for each in score_tresh:
#        roc_database[each] = {"pos" : 0,
#                              "neg" : 0}
#
#    for image_name, signs in images_with_sign.items():
#
#        for score_prob in score_tresh:
#
#            #choose one score probability and run detection
#            try:
#                image_path = os.path.join(dataset_path, image_name)
#                detector._thresholds[layers[net]] = score_prob
#                rois, _, _, _ = detector.detect(image_path, signs, layer=layers[net])
#            except ValueError:
#                #no box detected
#                continue
#
#            #save number of false positives for this one
#            detect_boxes = len(rois['pictures'])
#            print(detect_boxes)
#            roc_database[score_prob]['neg'] += detect_boxes
#
#
#            #check positives for each sign in image
#            for sign_info in signs:
#                sign_coordinates = np.array(sign_info["coordinates"]).astype(np.float32)
#
#                boxes_coordinates = rois['pictures'][:,:4]
#                pos, _ = detected(sign_coordinates, boxes_coordinates)
#
#                if len(pos) > 0:
#                    #only one, becouse we detect only one
#                    roc_database[score_prob]['pos'] += 1
#                    roc_database[score_prob]['neg'] -= len(pos)
#
#    print(roc_database)
#    #plot_roc(roc_database)

    
def iou_benchmark(model, net, threshold, dataset_info, dataset_path, class_mode):
    """
    Evaluate model on dataset based on intersection over union. 
    IOU tells us how much two boxes overlap. If detected bounding 
    box overlap with any GT bounding box with required certainty (threshold), 
    classify it as positive.
    Evaluating protocol is quite simple:
    1. Instantiate MTCNN and set model received as parameter. 
       Other two models should stay unchanged.
    2. Iterate over all images in dataset.
    3. For each image, run it through mtcnn and receive boxes
       from model we set.
    4. For each sign GT bounding box in single image calculate IOU
       with detected bounding boxes.

    Args:
        model: model to evaluate
        net: string, one of (first, second, third) 
        threshold: list of threshold, for each network in MTCNN
        dataset_info: dictionary containing GT info abou signs in dataset
        dataset_path: path to root folder of dataset
        class_mode: string, one of (binary, super, multi)
    Return:
        Dictionary that containg evaluation result.
    """

    sign_counter = 0
    detected_sign = 0
    box_counter = 0
    distribution = {
        'pos' : 0,
        'parts' : 0,
        'false' : 0
    }

    layers =  {
        "first" : 0,
        "second" : 1,
        "third" : 2
    }

    images_with_sign = dataset_info
    image_number = len(images_with_sign)

    detector = init_mtcnn(model, net, class_mode)

    # Iterate only on images with at least single sign
    for image_name, signs in images_with_sign.items():
        image_sign_count= len(signs)
        sign_counter += image_sign_count
        image_detected =0
        image_positives = []
        image_parts = []

        try:
            # Run detection.
            image_path = os.path.join(dataset_path, image_name)
            rois, _, _, _ = detector.detect(image_path, signs, layer=layers[net])
        except ValueError as e:
            # No box detected, so skip to the next one.
            print(e)
            continue
        box_counter += len(rois['pictures'])

        for sign_info in signs:
            class_number = sign_info["class"]
            super_class = sign_info["super-class"]
            sign_coordinates = np.array(sign_info["coordinates"]).astype(np.float32)

            boxes_coordinates = rois['pictures'][:,:4]
            pos, parts = detected(sign_coordinates, boxes_coordinates, threshold=threshold)

            image_positives.extend(pos)
            image_parts.extend(parts)

            # Remove duplicates
            image_positives = list(set(image_positives))
            image_parts = list(set(image_parts)) 

            if class_mode == "multi":
                for box_index in pos:
                    predicted_class_number = rois['pictures'][box_index][5].astype(np.int32)
                    if str(predicted_class_number) == str(class_number):
                        image_detected += 1
                        continue
            else:        
                if len(pos) != 0:
                    image_detected += 1

        # Store information about how many positive, parts and negative
        # bounding boxes model detected.
        pos_len = len(image_positives)
        parts_len = len(image_parts)
        false_pos_len = len(rois['pictures']) - pos_len - parts_len
        distribution['pos'] += pos_len
        distribution['parts'] += parts_len
        distribution['false'] += false_pos_len

        detected_sign += image_detected

    bench_info = {
        "model" : model,
        "iou_positive" : threshold,  
        "images" : image_number,
        "box_generated" : box_counter,
        "positive" : (distribution['pos'], round((100 * distribution['pos'] / box_counter),2)),  
        "false positive" :(distribution['false'], round((100 * distribution['false'] / box_counter),2)),  
        "parts" :(distribution['parts'], round((100 * distribution['parts'] / box_counter),2)),  
        "total_sign" : sign_counter,
        "detected" : detected_sign,
        "accuracy" : round((100 * detected_sign / sign_counter),4),
        "setup" : {
            "factor" : detector._factor,
            "thresholds" : detector._thresholds
        }
    }
    return bench_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Setup benchmark.")
    parser.add_argument('-p', dest='print_info', action='store_true', help='print information on standard output')
    parser.add_argument('bench_type', choices=['roc', 'iou', ""], help='what kind of bench we wanna')
    parser.add_argument('net', choices=['first', 'second', "third"], help='which net we want to test')
    parser.add_argument("model", help="benchmark will run on this model")
    parser.add_argument("dataset", help="path to dataset folder")
    parser.add_argument('classification', choices=['binary', 'super', "multi"], help='type of dataset')
    parser.add_argument("iou", nargs="?", default=0.7, type=float, help="threshold for positive detection")

    args = parser.parse_args()
    dataset_path = args.dataset
    training_info, validation_info, testing_info = detection_dataset_wrapper(dataset_path)

    if args.bench_type == "iou":

        bench_info = iou_benchmark(
            args.model, 
            net=args.net, 
            threshold=args.iou, 
            dataset_info=testing_info[0], 
            dataset_path=dataset_path, 
            class_mode=args.classification)

        if args.print_info:
            write_output(bench_info)

        #create storage file for benchmark information
        model_base_name = os.path.basename(args.model).split(".")[0]
        file_name = model_base_name + "_benchmark.json"
        
        save_json(data=bench_info, file_name=file_name)

    elif args.bench_type == "roc":
        roc_benchmark(
            args.model, 
            net=args.net, 
            threshold=args.iou, 
            dataset_info=validation_info[0], 
            dataset_path=dataset_path, 
            class_mode=args.classification)



