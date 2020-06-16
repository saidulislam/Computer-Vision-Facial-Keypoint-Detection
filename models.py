## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 5x5 square convolution kernel
        
        # input image 224x224
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (10, 220, 220)
        # after one maxpool layer, this becomes (10, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5) # 1 is the input channel here, 32 output - you can play with this number, 5 is the kernel
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, 
        # and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output Tensor for one image, will have the dimensions: (32, 106, 106)
        # after one maxpool layer, this becomes (32, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5) # 32 is the input from prev layer, 64 output - you can play with this number, 5 is the kernel
        self.conv2_bn = nn.BatchNorm2d(64)
        
        # output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output Tensor for one image, will have the dimensions: (64, 49, 49)
        # after one maxpool layer, this becomes (64, 24, 24)
        self.conv3 = nn.Conv2d(64, 128, 5) # 64 is the input from prev layer, 128 output - you can play with this number, 5 is the kernel
        self.conv3_bn = nn.BatchNorm2d(128)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        
        # 128 outputs from conv layer
        self.fc1 = nn.Linear(128*24*24, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        
        # 1024 from the previous fc layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        
        # 512 from the previous fc layer
        # final 136 points
        self.fc3 = nn.Linear(512, 136)
        
        # dropout with p=0.4
        self.dropout = nn.Dropout(p=0.4)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        #print(f'***** x = {x.shape}')
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        #print(f'***** x1 = {x.shape}')
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        #print(f'***** x2 = {x.shape}')
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        #print(f'***** x3 = {x.shape}')

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x)
        
        # final output
        return x
