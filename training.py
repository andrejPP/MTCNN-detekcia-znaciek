import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from nets import FirstNet, SecondNet, LastNet, check_mode
from func import load_model, save_model, Timer
from samplers import BasicSampler, BatchSamplerMulti
from dataset import SignDataset
from loss_tracker import LossTracker
from benchmark import iou_benchmark
import matplotlib.pyplot as plt
import argparse
import torch.multiprocessing as multi_proc
from copy import deepcopy


loss_names = ["full", "class", "reg"]

def compute_loss_mean(loss_dict):
    """
    Expect loss in dictionary with 3 keys, names of those keys 
    can be found in global variable "loss_names". Each of them 
    should have list of float values. Calculate mean for each 
    list.
    
    Return:
        Dictionary with the same keys as in "loss_dict" parameter
        and value is mean of those list.TODO
    """

    loss_mean = {}

    for loss_name in loss_names:
        if not isinstance(loss_dict[loss_name], list):
            raise RuntimeError("Value for key {} should be instance of list".format(loss_name))

        loss_list  = loss_dict[loss_name]
        if len(loss_list) > 0:
            loss_mean[loss_name] = sum(loss_list) / float(len(loss_list))
        else:
            loss_mean[loss_name] = 0
    
    return loss_mean


def compute_loss(class_out,
                 regresion_out,
                 labels,
                 coor,
                 class_loss_weight,
                 regresion_loss_weight,
                 device,
                 online_mining=False,
                 top_samp=0.7):
    """
    Using Cross entropy loss for classification and MSE loss for 
    regression evaluate wighted loss for this output of network.

    Args:
        class_out: classification output from network
        regresion_out: regression output from network
        labels: ground truth label for each sample in batch
        coor: ground truth coordination for each sample in batch
        class_loss_weight: weight for classification loss
        regresion_loss_weight: weight for regression loss
        device: gpu or cpu
        online_mining: bool if online mining should be used
        top_samp: what percentage of top samples should be used in 
            online mining

    Return:
       Classification, reggresion and weighted loss.
    """

    if online_mining == True:
        class_loss_fn = nn.CrossEntropyLoss(reduction='none')
        reg_loss_fn = nn.MSELoss(reduction='none')
    else:
        class_loss_fn = nn.CrossEntropyLoss()
        reg_loss_fn = nn.MSELoss()

    # Dont use part samples for classification loss calculation.
    usable_samples = np.where(labels.cpu() != -1)
    class_out = class_out[usable_samples].to(device)
    class_labels = labels[usable_samples].to(device)
    
    # Dont use negative samples for regression loss calculation.
    usable_samples = np.where(labels.cpu() != 0)
    regresion_out = regresion_out[usable_samples].to(device)
    coor = coor[usable_samples].to(device)

    # Calculate regression and classification loss. 
    class_loss = class_loss_fn(class_out, class_labels).to(device)
    reg_loss = reg_loss_fn(regresion_out, coor).to(device)

    if online_mining:
        class_loss = extract_best_samples(class_loss, top_samp, device)
        reg_loss = torch.mean(reg_loss, dim=1)
        reg_loss = extract_best_samples(reg_loss, top_samp, device)

    # Calculate wighted loss
    final_loss = class_loss*class_loss_weight + reg_loss*regresion_loss_weight

    # Check for NAN values.
    nan_exist = torch.isnan(final_loss).any()
    if nan_exist:
        raise RuntimeError("Loss function value is NAN.")
    return final_loss, class_loss, reg_loss

def extract_best_samples(tensor, top_samples, device):

    idx_sorted = torch.argsort(tensor, descending=True).to(device)
    size = round(len(idx_sorted)*top_samples)
    picked = tensor[idx_sorted[:size]]
    return torch.mean(picked).to(device)


def run_epoch(epoch: int,
              dataloader: DataLoader, 
              optimizer,
              scheduler, 
              config: dict, 
              device, 
              loss_tracker: LossTracker):
    """
    Run single training epoch.

    Args:
        epoch: epoch index
        dataloader: Dataloader instance 
        optimazer: instance of some optimizer from
            torch.optim library
        sheduler: instance of some sheduler from
            torch.optim.lr_scheduler library
        config: dictionary contains configuration details for this training 
            session
        device: cpu or gpu
        loss_trracker: LossTracker instance
    Return:
        Dictionary which contains losses for every single batch in this epoch.
    """

    running_loss = {loss_name : [] for loss_name in loss_names }
    epoch_loss = {loss_name : [] for loss_name in loss_names }
    timer = Timer()
    # How often (number of batches) print info about training.
    show_info = config['batches-info']
    
    # Check if online hard sample mining should be used.
    if 'ohsm' in config:
        ohsm = True
        ohsm_top = float(config['ohsm'])
    else:
        ohsm = False
        ohsm_top = 0

    timer.start()
    for index, data in enumerate(dataloader, 0):
        optimizer.zero_grad()

        batch, labels, coor = data
        batch = batch.to(device)
        labels = labels.to(device)
        coor = coor.to(device)

        # debug_batch(batch, batch_size, labels)
        # Run model with batch.
        class_map, box_regression = model(batch)

        if config['net'] == "first":
            num_class = check_mode(config["class-mode"])
            class_map = class_map.permute(0, 2, 3, 1).view(-1, num_class).to(device)
            box_regression = box_regression.permute(0, 2, 3, 1).view(-1, 4).to(device)

        final_loss, class_loss, reg_loss = compute_loss(class_map, box_regression, labels, coor,
                        config['classification_weight'],
                        config['regression_weight'],
                        device,
                        ohsm,
                        ohsm_top)

        for param_group in optimizer.param_groups:
            last_lr = param_group['lr']
        #loss_tracker.save_lr_history(last_lr, final_loss.item())
        
        final_loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss["full"].append(final_loss.item())
        running_loss["class"].append(class_loss.item())
        running_loss["reg"].append(reg_loss.item())

        if index % show_info == show_info-1:
            # Print info about training.
            avg_loss =  sum(running_loss['full']) / len(running_loss['full'])
            print(f"[{epoch + 1}, {index + 1}] loss: {avg_loss:.4f} lr: {last_lr:.3f} time: {timer.round():.4f} sec")

            extend_dict_keys(dest_dict=epoch_loss, source_dict=running_loss)
            running_loss = {loss_name : [] for loss_name in loss_names }

        extend_dict_keys(dest_dict=epoch_loss, source_dict=running_loss)

    return epoch_loss


def train(model, config, loss_tracker):
    """
    Setup and run training session for neural network.

    Args:
        model: neural network model that is trained
        config: dictionary contains configuration details for this training 
            session
        loss_tracker: LossTracker instance used for tracking training details
    """

    device = config['device']

    transform = transforms.Compose(
        [transforms.ColorJitter(brightness=config['b'],
                                contrast=config['c'],
                                saturation=config['s'],
                                hue=config['h']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Choose optimizer.
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['base_lr'], momentum=config['momentum'])
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['base_lr'])

    # Create sheduler.
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config['base_lr'], max_lr=config['max_lr'], step_size_up=config['cycle'])

    # Setup DataLoader.
    dataset =  SignDataset(config['dataset'], net=config['net'], classification=config["class-mode"], transform=transform)
    batch_sampler = setup_sampler(dataset, batch_size=config['batch_size'])
    dataloader = DataLoader(dataset, pin_memory=True,  batch_sampler=batch_sampler)

    if "batches-info" not in config:
        config["batches-info"] = len(dataloader)

    for epoch in range(config['epochs_num']):
        # Train single epoch.
        train_loss_dict = run_epoch(epoch, dataloader, optimizer, scheduler, config, device, loss_tracker)
        loss_mean = compute_loss_mean(train_loss_dict)

        # Add tracked losses from this training epoch.
        loss_tracker.add_loss(loss_mean["full"], loss_mean["class"], loss_mean["reg"], "training")
       
        # Save model after training epoch.
        save_model(model, config['model'])

        if epoch % config['validate'] == config['validate'] - 1:
            bench_result = run_benchmark(config, 0.65)
            loss_tracker.add_bench_data(bench_result, model, epoch + 1)

            # Run validation and save the result.
            valid_loss_dict = test(model, config, loss_tracker)
            v_loss_mean = compute_loss_mean(valid_loss_dict)
            loss_tracker.add_loss(v_loss_mean["full"], v_loss_mean["class"], v_loss_mean["reg"], "validation")
            
            print(f"Current loss on validation dataset is {v_loss_mean['full']:.4f}")
            print(f"Benchmark accuracy is {bench_result['accuracy']:.4f} with {bench_result['box_generated']} boxes") 
            print(f"Validation loss is {v_loss_mean['full']/loss_mean['full']:.4f} times bigger then training loss")
            
            diff = v_loss_mean["full"] / loss_mean["full"]
            if ('epsilon' in config) and (diff > config['epsilon']):
                raise RuntimeError("Epsilon value reached, threshold")

def test(model, config, loss_tracker):
    """
    Run trained model on part of the dataset used for validation.

    Args:
        model: neural network model that is validated
        config: dictionary contains configuration details for this validation
        loss_tracker: LossTracker instance used for tracking validation details
    Return:
        Dictionary with all losses.
    """

    mode = "test"
    device = config['device']
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset =  SignDataset(config['dataset'], 
                        net=config['net'], 
                        classification=config["class-mode"], 
                        mode=mode, 
                        transform=transform)

    batch_sampler = setup_sampler(dataset, batch_size=config['batch_size'])
    mini_batch_count = len(batch_sampler)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    correct = 0
    total = 0
    running_loss = {loss_name : [] for loss_name in loss_names }
    with torch.no_grad():
        for data in dataloader:

            batch, labels, coor = data
            batch = batch.to(device)
            labels = labels.to(device)

            # Run model with batch.
            class_map, box_regression = model(batch)

            if config['net'] == "first":
                num_class = check_mode(config["class-mode"])
                class_map = class_map.permute(0, 2, 3, 1).view(-1, num_class).to(device)
                box_regression = box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4).to(device)

            usable_labels = np.where(labels != -1)
            _, predicted = torch.max(class_map[usable_labels].data, 1)
            predicted = predicted.view(1, -1)[0]
            total += labels[usable_labels].size(0)
            correct += (predicted == labels[usable_labels]).sum().item()

            final_loss, class_loss, reg_loss = compute_loss(class_map, box_regression, labels, coor,
                         config['classification_weight'],
                         config['regression_weight'],
                         device)

            running_loss["full"].append(final_loss.item())
            running_loss["class"].append(class_loss.item())
            running_loss["reg"].append(reg_loss.item())

    print(f"Accuracy of the network on validation dataset: {100*correct/total:.2f}")
    return running_loss


def setup_sampler(dataset, batch_size: int) -> BatchSamplerMulti:
    """ 
    Create sampler with proper ratio of positive, negative and part
    samples in each batch.
    """

    class_distribution = dataset.dataset_info()

    part = class_distribution['parts']
    neg = class_distribution['negatives']
    pos = class_distribution['positives']

    neg_sampler = BasicSampler(start=neg[0], end=neg[1])
    pos_sampler = BasicSampler(start=pos[0], end=pos[1])
    part_sampler = BasicSampler(start=part[0], end=part[1])
    samplers = {neg_sampler: 0.60,
                pos_sampler: 0.2,
                part_sampler: 0.2}

    return BatchSamplerMulti(samplers=samplers, batch_size=batch_size)


def run_benchmark(config: dict, iou: float):

    try:
        dataset_path = config["bench_dataset"]
    except KeyError:
        # Try to use default path.
        dataset_path = "./FullIJCNN2013/"

    from datasets_wrapper import detection_dataset_wrapper
    _,_, testing_info = detection_dataset_wrapper(dataset_path)

    result = iou_benchmark(
        model=config['model'], 
        net=config['net'], 
        threshold=iou, 
        dataset_info=testing_info[0], 
        dataset_path=dataset_path, 
        class_mode=config["class-mode"])
    return result


def extend_dict_keys(dest_dict: dict, source_dict: dict):
    """
    Take two dictionaries and extend one by keys and values from the other one. 
    Args:
        source_dict: the source dictionary, which should not be extended
        dest_dict: the destination dictionary, which should be extended
            by keys and values form source_dict
    """

    for key, data in source_dict.items():
        if (isinstance(data, list) and 
            key in dest_dict.keys() and isinstance(dest_dict[key], list)):
            dest_dict[key].extend(data)


def debug_batch(batch, batch_size, labels):
    """
    Debug function printing simple details about single batch.
    """
    idx = np.where(labels == 0)
    perc = float(100/batch_size)*len(idx[0])
    print(f"Number of negative samples: {len(idx[0])} what is {perc} %.")

    idx = np.where(labels == 1)
    per = float(100/batch_size)*len(idx[0])
    print(f"Number of positive samples: {len(idx[0])} what is {perc} %.")

    idx = np.where(labels == 2)
    per = float(100/batch_size)*len(idx[0])
    print(f"Number of part samples: {len(idx[0])} what is {perc} %.")

    print(f"Batch size: {batch.size()[0]}")


def load_config(config_path):
    from func import load_json
    config = load_json(config_path)
    # Add to the config name of configuration file without extension
    config['name'] = os.path.basename(config_path).split(".")[0]
    config = check_model(config)
    return config


def check_model(config):
    """
    Check if the path to the model is set up in config. 
    If it is not, use configuration name as model name and 
    create proper path.
    """
    
    if "model" not in config:
        config['model'] = "./models/" + config['name'] + ".pt"
    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load config file.')
    parser.add_argument('config_file', help='json config file for training')
    parser.add_argument('--use-cuda', dest='cuda', action='store_true', help='activate CUDA if available')
    args = parser.parse_args()
    config = load_config(args.config_file)
    name = os.path.splitext(os.path.basename(args.config_file))[0]
    loss_tracker = LossTracker(config)

    # Based on one config parameter and availability of gpu choose training device (cpu/gpu)
    if args.cuda and torch.cuda.is_available():
        dev = torch.device('cuda')
        config['device'] = dev
    else:
        dev = torch.device('cpu')
    config['device'] = dev
    print(f" Running on {dev}.")

    # Based on config parameter choose type of net, that should be trained.
    if config['net'] == "first":
        model = FirstNet(config['class-mode'], channels=3).to(dev)
    elif config['net'] == "second":
        model = SecondNet(config['class-mode'], channels=3).to(dev)
    elif config['net'] == "third":
        model = LastNet(config['class-mode'], channels=3).to(dev)
    else:
        raise ValueError("Unknown type of net->", config['net'])
    load_model(model, config['model'], mode="train")

    try:
        train(model, config, loss_tracker)
    finally:
        save_model(model, config['model'])
        loss_tracker.write_to_file()