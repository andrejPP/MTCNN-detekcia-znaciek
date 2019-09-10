########################################################
#      
#      Autor: Andrej Paníček           
#      Posledná zmena: 2.3.2019
#      Popis: Model obsahuje implementáciu sietí P-Net, 
#      R-Net, O-Net
#
########################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary


#NETS conv2d(in_channels, out_channels, kernel_size)
#     MaxPool2d(kernel_size, stride)


def check_mode(class_mode):
    classification = {"binary": 2, "super": 6, "multi": 44}
    if class_mode not in classification:
        raise ValueError("Unknown classification mode ->", class_mode) 
    return classification[class_mode]


class FirstNet(nn.Module):

    def __init__(self, classification_mode, channels):
        super(FirstNet, self).__init__()
        num_class = check_mode(classification_mode)

        self.main_model = nn.Sequential(
            #input paramaters -> channels, output, kernel size
            nn.Conv2d(channels,10,3),
            nn.PReLU(),
            # input parameters-> kernel, stride
            nn.MaxPool2d(2,2),
            nn.Conv2d(10,16,3),
            nn.PReLU(),
            nn.Conv2d(16,32,3),
            nn.PReLU()
            )
        self.clasifier = nn.Conv2d(32, num_class, 1)
        self.bounding_box_regression = nn.Conv2d(32,4, 1)

    def forward(self, data):
        output = self.main_model(data)
        #use softmax on dimension no. 1 which is channel dimension
        class_map = self.clasifier(output)
        # class_map = F.softmax(self.clasifier(output), dim=1)
        boxes_reg = self.bounding_box_regression(output)

        return class_map, boxes_reg

    def stride(self):
        pass

    
class SecondNet(nn.Module):

    def __init__(self, classification_mode, channels):
        super(SecondNet, self).__init__()
        num_class = check_mode(classification_mode)

        self.convolutional = nn.Sequential(
            nn.Conv2d(channels,28,3),
            nn.PReLU(),
            nn.MaxPool2d(3,2,ceil_mode=True),
            nn.Conv2d(28,48,3),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(48,64,2),
            nn.PReLU()
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(3*3*64,128),
            nn.PReLU()
        )
        self.fc1 = nn.Linear(128,num_class)
        self.fc2 = nn.Linear(128,4)

    def forward(self, data):
        output = self.convolutional(data)
        output = output.view(-1,3*3*64)
        output = self.fully_connected(output)
        class_map = self.fc1(output)
        boxes_reg = self.fc2(output)

        return class_map, boxes_reg

class LastNet(nn.Module):

    def __init__(self, classification_mode, channels):
        super(LastNet, self).__init__()
        num_class = check_mode(classification_mode)

        self.convolutional = nn.Sequential(
            nn.Conv2d(channels,32,3),
            nn.PReLU(),
            nn.MaxPool2d(3,2, ceil_mode=True),
            nn.Conv2d(32,64,3),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(64,64,3),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,2),
            nn.PReLU()
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(3*3*128,256),
            nn.Dropout(0.25),
            nn.PReLU()
        )
        self.fc1 = nn.Linear(256,num_class)
        self.fc2 = nn.Linear(256,4)

    def forward(self, data):
        output = self.convolutional(data)
        output = output.view(-1,3*3*128)
        output = self.fully_connected(output)
        class_map = self.fc1(output)
        boxes_reg = self.fc2(output)

        return class_map, boxes_reg
