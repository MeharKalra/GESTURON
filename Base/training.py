# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 18:43:10 2022

@author: Mehar Kalra
"""

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2 as cv
import time
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        #print(size)
        num_features = 1
        for s in size:
            num_features = num_features*s
        return num_features


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
from torchvision import datasets
data_f1 = datasets.ImageFolder(root = 'train_data/',transform = transform)
folderL = DataLoader(data_f1,batch_size = 20,shuffle = True)
images,labels = iter(folderL).next()
print(labels)


s = time.time()
torch.manual_seed(0)
mini_batch = 20
loss_values = []
for epoch in range(30):  # loop over the dataset multiple times

    #running_loss = 0.0
    
    for i, data in enumerate(folderL, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # make the parameter gradients zero
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        #running_loss += loss.item()
        if i % 10 == 0:    # print every 200 mini-batches
             print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, loss.item()))
        loss_values.append(loss.item())
        #running_loss = 0.0
e = time.time()       
plt.plot(loss_values)

print('Finished Training')
print('Training time is:',(e-s)/60,'minutes')
correct = 0
total = 0
with torch.no_grad():
    for data in folderL:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the train images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in folderL:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(18):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        str(i), 100 * class_correct[i] / class_total[i]))
    
data_f2 = datasets.ImageFolder(root = 'test_new',transform = transform)
folderTest = DataLoader(data_f2,batch_size = 5,shuffle = False)

correct = 0
total = 0
with torch.no_grad():
    for data in folderTest:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(predicted)

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in folderTest:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(1):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        str(i), 100 * class_correct[i] / class_total[i]))
    
#PATH = './my_own_three_class_net.pth'
#torch.save(net.state_dict(), PATH)