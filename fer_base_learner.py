import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import random
import copy
import gc
import numpy
import argparse
from sklearn.utils import shuffle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
#from model import Net
from dataset_object import Dataset as D
from PIL import Image
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: 
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout(0.4, inplace=True))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.4),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def train(sched, net, trainDataLoader, optimizer, criterion, validDataLoader, device, n_classes):
#def train(epochs, net, trainDataLoader, optimizer, criterion, device, n_classes):

    net.train()
    train_loss = 0
    iteration = 0
    correct, total, p_correct, p_total = 0, 0, 0, 0
    correct_array = []
    loss_value_array = []
    patience = 5 
    best = 10000
    best_model = copy.deepcopy(net)
    #for e in range (args.epochs):
    #    train_loss = 0
    #    valid_loss = 0
    for ix, sample in enumerate(trainDataLoader):
        #inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = sample['X'], sample['Y']
        #if torch.cuda.is_available():

        inputs, targets = inputs.to(device), targets.to(device)
        #targets = targets.type(torch.float)

        outputs = net (inputs)
        outputs = outputs.to(device)

        #outputs = outputs.type(torch.float)
        outputs = torch.softmax(outputs,1)
        #   outputs = torch.argmax(outputs,1)

        binary_targets = torch.zeros((len(inputs), n_classes))
        binary_targets = binary_targets.to(device)

        index = 0
        with torch.no_grad():
            for t in targets:
                binary_targets[index, t.item()] = 1
                index += 1

        loss = criterion(outputs, binary_targets)
            
        if ( (ix+1)%100==0):
            print("Loss function value: ")
            print(str(loss.item()))

            
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()
        train_loss += loss.item()
        optimizer.zero_grad()
        sched.step()

    net.eval()
    valid_loss = 0    
    for sample in validDataLoader:

        inputs, targets = sample['X'], sample['Y']
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        out = net (inputs)
        out = out.to(device)

        #out = outputs.type(torch.float)

        out = torch.softmax(out, 1)

        index = 0

        binary_targ = torch.zeros((len(inputs), n_classes))

        binary_targ = binary_targ.to(device)

        for t in targets:
            binary_targ[index, t.item()] = 1
            index += 1

        new_target = binary_targ.detach().cpu().numpy()
        new_outputs = out.detach().cpu().numpy()

        index = 0

        for b in new_target:
            a = numpy.argmax(b)
            pred = numpy.argmax(new_outputs[index, :])
            
            if args.primary_class.find(str(a.item())) == -1 :
                p_total += 1
                if (a.item()==pred):
                    p_correct += 1

            else:
                total += 1
                if (a.item()==pred):
                    correct += 1

            index += 1

            

        loss = criterion(out, binary_targ)

        valid_loss += loss.item()

    print(correct,total,p_correct,p_total) 
    
    print('Accuracy of the network on trained classes:')
    print(str(float(100 * correct / total)))
    print('Accuracy of the network on non-trained classes:')
    print(str(float(100 * p_correct / p_total)))
    
    # calculate average losses
    if  valid_loss < ( best - 1e-4):
        print("Saving best model . . . ")
        best_model = copy.deepcopy(net)
        torch.save(best_model.state_dict(),"best_model_0")
        fails = 0
        best = valid_loss
    else:
        fails = fails + 1
    print("Epoch: ",e," \n Train_loss: ",train_loss/len(trainDataLoader.sampler), "\n Valid Loss: ",valid_loss/(len(validDataLoader.sampler)))    
    #if fails > patience:
    #    print("Early Stopping. Valid hasn't improved for {}".format(patience))
    
def test(net, testDataLoader, device, primary_classes, n_classes):
    print("Test /n") 
    net.eval()
    correct, total, p_correct, p_total = 0, 0, 0, 0

    for sample in testDataLoader:

        inputs, targets = sample['X'], sample['Y']
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)

       

        binary_targets = torch.zeros((len(inputs), n_classes))

        binary_targets = binary_targets.to(device)
        
        index = 0

        for t in targets:
            binary_targets[index, t.item()] = 1
            index += 1

        #_, pred = torch.max(outputs, 1)

        index = 0

        new_target = binary_targets.detach().cpu().numpy()
        new_outputs = outputs.detach().cpu().numpy()

        for b in new_target:
            a = numpy.argmax(b)
            pred = numpy.argmax(new_outputs[index, :])
            #total + =1
            #if (a.item()==pred):
            #    correct += 1
            
            if args.primary_class.find(str(a.item())) == -1 :
                p_total += 1
                if (a.item()==pred):
                    p_correct += 1

            else:
                total += 1
                if (a.item()==pred):
                    correct += 1

            index += 1
    
    print(correct,total,p_correct,p_total)
    print('Accuracy of the network on trained classes:')
    print(str(float(100 * correct / total)))
    print('Accuracy of the network on non-trained classes:')
    print(str(float(100 * p_correct / p_total)))


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--primary_class',type=str, default=0,
                    help='class(es) the base learner is mainly trained on')
parser.add_argument('--model_name', type=str, default='model_',
                    help='Model name for saving - will not save model if name is None')
parser.add_argument('--lr', type=float,default=0.95,
                    help='learning rate')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--batch_size', type=int,default=64,
                    help='batch size')
parser.add_argument('--epochs', type=int,default=100,
                    help='number of epochs')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tfms = tt.Compose([tt.RandomCrop(48, padding=4, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor()
                         ])
valid_tfms = tt.Compose([tt.ToTensor()])

train_dataset = ImageFolder('data/train', tt.ToTensor())
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64)

mean, std = get_mean_and_std(train_dataloader)

train_ds = ImageFolder('data/train', train_tfms)
valid_ds = ImageFolder('data/test', valid_tfms)

base_learner_train_data = []
base_learner_train_label = []
base_learner_test_valid_data = []
base_learner_test_valid_label = []

index, others, n = 0, 0, 0

if args.primary_class == "16": #3500
    a = 3 # 1/4
elif args.primary_class == "01" or args.primary_class == "12": #4500
    a = 6 # 1/3
elif args.primary_class == "14" or args.primary_class == "15": #5400
    a = 7 # 40%
elif args.primary_class == "06" or args.primary_class=="26": #7000
    a = 5 # 55%
elif args.primary_class == "02" or args.primary_class == "46" or args.primary_class=="56" or args.primary_class == "13": #8000
    a = 6 # 60%
elif args.primary_class == "04" or args.primary_class == "05" or args.primary_class == "24" or args.primary_class == "25": #9000
    a = 7 #70%
elif args.primary_class == "36" or args.primary_class == "45": #10000
    a = 8 #80%
elif args.primary_class == "03" or args.primary_class == "23": #11000
    a = 13 #90%
elif args.primary_class == "34" or args.primary_class == "35": #12000
    a = 14 #100%

    
# 3995 436 4097 7215 4965 4830 3171

for l in train_ds:
    data = l[0].numpy()
    label = l[1]
    dice = random.randint(1,20)
    if (args.primary_class.find(str(label)) != -1 or dice<=a):
        if args.primary_class.find(str(label)) != -1:
            n += 1
        else:
            others += 1    
        base_learner_train_data.append(data)
        base_learner_train_label.append(label)
    #else:
    #    base_learner_test_valid_data.append(data)
    #    base_learner_test_valid_label.append(label)


print("Number of samples of particular class ",n)
print("Number of samples of all other classes ",others)

train_data = np.array(base_learner_train_data)
train_labels = np.array(base_learner_train_label)

#test_data = np.array(base_learner_test_valid_data)
#test_labels = np.array(base_learner_test_valid_label)

train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2)
train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.2)

trainSignData = D(train_data, train_labels,"rgb")
trainDataLoader = torch.utils.data.DataLoader(trainSignData, shuffle=True, batch_size=args.batch_size)

validSignData = D(valid_data, valid_labels,"rgb")
validDataLoader = torch.utils.data.DataLoader(validSignData, shuffle=True, batch_size=args.batch_size)

testSignData = D(test_data, test_labels,"rgb")
testDataLoader = torch.utils.data.DataLoader(testSignData, shuffle=True, batch_size=args.batch_size)

model = ResNet9(3, 7).to(device)

if torch.cuda.is_available():
    print('CUDA is available!  Training on GPU ...\n')
    model.cuda()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
max_lr = 0.001
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=args.epochs,
                                                steps_per_epoch=len(trainDataLoader))
n_classes = 7

for e in range(args.epochs):
    train(sched, model, trainDataLoader, optimizer, criterion, validDataLoader,device,n_classes) 

test(model, testDataLoader, device, args.primary_class, n_classes)

path = "spec_models/"

torch.save(model.state_dict(), "spec_models/model_"+args.primary_class)
