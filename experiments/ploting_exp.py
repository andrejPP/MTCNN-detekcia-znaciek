#!/usr/bin/env python3

import sys
sys.path.append("../")

import os
import matplotlib.pyplot as plt
import numpy as np
from func import load_json
#from search_result import only_dirs
import argparse

def extract_loss(data, mode):

    x = []
    full = []
    regression = []
    classification = []
    for epoch, loses in data[mode].items():
        x.append(epoch)
        full.append(loses["full loss"])
        regression.append(loses["regresion loss"])
        classification.append(loses["classification loss"])
    
    return x, full, classification, regression

def plot_compare_loss(args):

    for index, each in enumerate(args):
        plt.plot(each[1], label=index)

    plt.legend()
    #plt.savefig("{}.pdf".format("porovnanie_dat_2"), format="pdf")
    plt.show()

def plot_multiple_loss( *args):

    reg_val = []
    reg_tra = []
    for i in args[3]:
        reg_tra.append(i)
    for i in args[7]:
        reg_val.append(i)

    print(len(args))
    plt.plot(args[1], "r", label="tréning")
    #plt.plot(args[2], "b")
    #plt.plot(reg_tra, "g")
    plt.plot(args[5], "b", label="validácia")
    #plt.plot(args[6], "b")
    #plt.plot(reg_val, "y")
    plt.xlabel("Počet iterácií")
    plt.ylabel("Chyba")
    plt.legend()
    plt.savefig("{}.pdf".format("pnet_loss"), format="pdf")
    plt.show()

def plot_range_lr(*args):

    plt.plot(args[0], args[2], "b")
    plt.show()

    plt.plot(args[0], args[1], "b")
    plt.show()

def plot_accuracy(*args):

    plt.plot(args[0], "b")
    plt.show()

def plot_roc(data):

    import matplotlib.pyplot as plt

    data = [
        {1: {'pos': 53, 'neg': 1}, 0.95: {'pos': 201, 'neg': 56}, 0.9: {'pos': 206, 'neg': 73}, 0.85: {'pos': 207, 'neg': 83}, 0.8: {'pos': 209, 'neg': 97}, 0.75: {'pos': 209, 'neg': 113}, 0.7: {'pos': 209, 'neg': 126}, 0.65: {'pos': 210, 'neg': 134}, 0.6: {'pos': 212, 'neg': 147}, 0.55: {'pos': 213, 'neg': 158}, 0.5: {'pos': 213, 'neg': 173}, 0.45: {'pos': 213, 'neg': 184}, 0.4: {'pos': 213, 'neg': 196}, 0.35: {'pos': 213, 'neg': 214}, 0.3: {'pos': 213, 'neg': 231}, 0.2: {'pos': 213, 'neg': 289}, 0.15: {'pos': 213, 'neg': 331}, 0.1: {'pos': 213, 'neg': 404}, 0.05: {'pos': 213, 'neg': 573}, 0: {'pos': 213, 'neg': 43193}},
        {1: {'pos': 154, 'neg': 24}, 0.95: {'pos': 216, 'neg': 4199}, 0.9: {'pos': 216, 'neg': 5069}, 0.85: {'pos': 216, 'neg': 5696}, 0.8: {'pos': 216, 'neg': 6159}, 0.75: {'pos': 216, 'neg': 6573}, 0.7: {'pos': 216, 'neg': 6976}, 0.65: {'pos': 216, 'neg': 7364}, 0.6: {'pos': 216, 'neg': 7766}, 0.55: {'pos': 216, 'neg': 8107}, 0.5: {'pos': 216, 'neg': 8447}, 0.45: {'pos': 216, 'neg': 8859}, 0.4: {'pos': 216, 'neg': 9265}, 0.35: {'pos': 216, 'neg': 9678}, 0.3: {'pos': 216, 'neg': 10121}, 0.2: {'pos': 216, 'neg': 11181}, 0.15: {'pos': 216, 'neg': 11984}, 0.1: {'pos': 216, 'neg': 12993}, 0.05: {'pos': 216, 'neg': 14735}, 0: {'pos': 216, 'neg': 43127}},
        {1: {'pos': 85, 'neg': 0}, 0.95: {'pos': 213, 'neg': 102}, 0.9: {'pos': 214, 'neg': 127}, 0.85: {'pos': 215, 'neg': 141}, 0.8: {'pos': 215, 'neg': 160}, 0.75: {'pos': 216, 'neg': 171}, 0.7: {'pos': 216, 'neg': 186}, 0.65: {'pos': 216, 'neg': 200}, 0.6: {'pos': 216, 'neg': 212}, 0.55: {'pos': 216, 'neg': 228}, 0.5: {'pos': 216, 'neg': 244}, 0.45: {'pos': 216, 'neg': 256}, 0.4: {'pos': 216, 'neg': 274}, 0.35: {'pos': 216, 'neg': 297}, 0.3: {'pos': 216, 'neg': 320}, 0.2: {'pos': 216, 'neg': 388}, 0.15: {'pos': 216, 'neg': 443}, 0.1: {'pos': 216, 'neg': 531}, 0.05: {'pos': 216, 'neg': 725}, 0: {'pos': 216, 'neg': 43067}}
    ]
    num = ["výstup P-Net", "NH", "NH + výstup P-Net"]        
    ff = ["b", "c", "r"]
    #data = [#        {1: {'pos': 88, 'neg': 14}, 0.95: {'pos': 216, 'neg': 2425}, 0.9: {'pos': 216, 'neg': 3000}, 0.85: {'pos': 216, 'neg': 3422}, 0.8: {'pos': 216, 'neg': 3775}, 0.75: {'pos': 216, 'neg': 4099}, 0.7: {'pos': 216, 'neg': 4362}, 0.65: {'pos': 216, 'neg': 4646}, 0.6: {'pos': 216, 'neg': 4912}, 0.55: {'pos': 216, 'neg': 5188}, 0.5: {'pos': 216, 'neg': 5471}, 0.45: {'pos': 216, 'neg': 5745}, 0.4: {'pos': 216, 'neg': 6052}, 0.35: {'pos': 216, 'neg': 6360}, 0.3: {'pos': 216, 'neg': 6704}, 0.2: {'pos': 216, 'neg': 7569}, 0.15: {'pos': 216, 'neg': 8153}, 0.1: {'pos': 216, 'neg': 9003}, 0.05: {'pos': 216, 'neg': 10444}, 0: {'pos': 216, 'neg': 43129}}
    #]
    counter = 0
    for each in data:
        total = 216
        pos = []
        neg = []
        for tresh, _ in each.items():
            new = each[tresh]['pos'] / total
            pos.append(new)
            neg.append(each[tresh]['neg'])

        plt.plot(neg, pos)
        counter += 1

    plt.xlabel("Falošne pozitívne")
    plt.ylabel("Pomer úspešne zachytených značiek")
    plt.savefig("{}.pdf".format("rnet_roc"), format="pdf")
    plt.legend()
    plt.show()

#plot_roc("nic")

def plot_compare(args):

    new = [] 
    for each in args:
        new.append(each[:30])

    #num = ["4", "3", "2", "1"]        
    num = ["výstup P-Net", "NH", "NH + výstup P-Net"]        

    for index, each in enumerate(new):
        plt.plot(each, label=num[index])
        #plt.plot(each)

    plt.xlabel("Počet iterácii")
    plt.ylabel("Ǔspešnosť zachytených značiek v %")
    plt.legend()
    plt.savefig("{}.pdf".format("rnet_recall"), format="pdf")
    plt.show()

def plot_false_pos(*args):

    print(args[0])
    for index, each in enumerate(args[0],0):
        plt.plot(each, label=str(index))
        #plt.savefig("{}.pdf".format("pnet_recall"), format="pdf")
    plt.legend()
    plt.show()


def get_data_json(path):
    """
    path - except job folder
    """
    data_path = os.path.join(path,"outputs_folder/training_data.json")
    return load_json(data_path)


def extract_benchmark(data):

    acc = []
    benchmark = data['benchmark']
    for each in  benchmark:
        acc.append(each['accuracy'])

    return acc

def false_pos(data): 

    generated = []
    benchmark = data['benchmark']
    for each in  benchmark:
        generated.append(each['false positive'][0])

    return generated



def extact_lr_history(data):

    lr_history = data['LR range history']
    last_lr = 0
    x = []
    y = []
    z = []
    for each in lr_history:
        if each['lr'] > last_lr:
            last_lr = each['lr']
            x.append(each['lr'])
            y.append(each['loss'])
            if 'acc' in each:
                z.append(each['acc'])

    return x, y, z

def browser_jobs(path):

    #jobs = only_dirs(path)
    return jobs

def average_over(array,size):

    new_array = []

    total_sum = 0
    for index, x in enumerate(array, 1):
        total_sum += x
        if index % size == 0:
            new_array.append(total_sum/size)
            total_sum = 0

    return new_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', choices=['range', 'loss', 'compare', "false", "cp_loss"])
    parser.add_argument('loss_directory', nargs="*")
    args = parser.parse_args()
    if args.data == "loss":
        data = get_data_json(args.loss_directory[0])
        train_loss = extract_loss(data, 'training')
        valid_loss = extract_loss(data, 'validation')
        plot_multiple_loss(*train_loss, *valid_loss )
    elif args.data == "cp_loss":
        cp_list =  []

        for each  in args.loss_directory:
            data = get_data_json(each)

            train_loss = extract_loss(data, 'training')
            valid_loss = extract_loss(data, 'validation')
            cp_list.append(valid_loss)

        plot_compare_loss(cp_list)
 

    elif args.data == "range":
        data = get_data_json(args.loss_directory[0])
        lr, loss, acc = extact_lr_history(data)
        for index, i in enumerate(lr):
            print(i)
            if index == 10:
                break
        lr = average_over(lr, 1)
        loss = average_over(loss, 1)
        acc = average_over(acc, 1)
        plot_range_lr(lr, loss, acc)

    elif args.data == "compare":
        acc_list =  []

        for each  in args.loss_directory:
            data = get_data_json(each)
            acc = extract_benchmark(data)
            acc_list.append(acc)

        plot_compare(acc_list)
    
    elif args.data == "false":

        if len(args.loss_directory) == 0:
            data = get_data_json(args.loss_directory[0])
            negative = false_pos(data)
            plot_false_pos(negative)
        else:
            some_list =  []

            for each  in args.loss_directory:
                data = get_data_json(each)
                negative = false_pos(data)
                some_list.append(negative)
            
            plot_false_pos(some_list)




