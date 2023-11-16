import argparse
import copy
import time
import tarfile
import torch
import torchvision.models
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np
import perturbations.perturbations
from torchvision.datasets import ImageFolder
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
import time
from dataset_object import Dataset as D
from itertools import permutations
import matplotlib.pyplot as plt
#import keras
import cv2
import warnings
import sys
#import ktrain
#from ktrain import vision as vis
import torchvision.models as models


def init_weights(net, init_type='normal', init_gain=0.02):

    """
    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform(m.weight.data, a=0, mode='fan_in')
                init.kaiming_uniform(m.bias.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
                init.orthogonal_(m.bias.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix;
                                                   # only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
    print('initialize network with %s' % init_type)

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

# Selection score predictor
'''
class SelectNet(nn.Module):
    def __init__(self, n_models):
        super(SelectNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(774400, 1000)
        self.fc2 = nn.Linear(1000, n_models)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x) # from here
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = x #F.log_softmax(x, dim=1)   
        return output
'''
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

def test(selection_net, age_model, device, test_loader,args):
    #model.eval()
    test_loss = 0
    correct = 0
    C = args.c
    def batch_knapsack(scores): #64000*55
        indices = torch.topk(scores, C).indices #64000*21
        choice = torch.zeros_like( scores )

        #somma = torch.sum(scores(1,))


        choice.scatter_(1,indices,torch.ones(indices.shape,device=device))
        return choice

    knapsack_layer = perturbations.perturbations.perturbed_special(batch_knapsack,                                                                                 num_samples=1000,
                                                                   sigma=0.1,
                                                                   noise='normal',
                                                                   batched=True,
                                                                   device=device,
                                                                   hard_fwd=True )
    correct, total, p_correct, p_total, nIm = 0, 0, 0, 0, 0
    iteration = 0
    with torch.no_grad():
        for sample in test_loader:

            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # probably not perfect wrt tensor orientation
            dim = target.shape[0]
            age_predictions = torch.zeros(len(age_model), dim, 7).to(device)
            # gender_predictions = torch.zeros(len(age_model), dim, 2)
            # race_predictions = torch.zeros(len(age_model), dim, 5)

            #age_predictions = torch.stack([torch.softmax(m(data),1) for m in
            #                               age_model])  # n_models*batch_size*_classes   #majority_voter(model,data)
            #age_predictions = torch.stack( [torch.softmax(m(data),1) for m in age_model] )
            age_predictions = torch.stack([m(data) for m in age_model])
            # gender_predictions = torch.stack( [m(data) for m in gender_model] )
            # race_predictions = torch.stack( [m(data) for m in race_model] )
            selection_vals = selection_net(data)  # batch_size*n_models
            # selection_vals.requires_grad_()
            selection_vals = torch.nn.functional.normalize(selection_vals)
            #selection_vals = torch.nn.Relu(selection_vals)
            if (args.injection):
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - torch.topk(age_predictions, 2, 2).values[:, :,1])
                selections = knapsack_layer(selection_vals*diff.T)
            else:
                selections = knapsack_layer(selection_vals)  # batch_size*n_models

            #selection_vals = torch.nn.functional.normalize(selection_vals)  # before I applied L2 normalization - torch.nn.functional.normalize(selection_vals)
            if (args.weight_pred):
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - torch.topk(age_predictions, 2, 2).values[:, :, 1]).repeat(7, 1, 1)
                diff = torch.permute(diff, (1, 2, 0))
                age_predictions = age_predictions * selections.repeat(7, 1, 1).T * diff #selection_vals.repeat(7,1,1).T
            else:
                age_predictions = age_predictions * selections.repeat(7, 1, 1).T

            majority_vote = torch.zeros(dim, 7).to(device)

            if (args.apply_sum):
                majority_vote = torch.sum(age_predictions, 0)
            else:
                majority_vote = torch.mean(age_predictions, 0)

            age_binary_target = torch.zeros((dim, 7))

            age_binary_target = age_binary_target.to(device)

            index = 0
            p = 0
            for t in target:
                age_binary_target[index, t.item()] = 1
                index += 1


            index = 0
            age_correct, gender_correct, race_correct = 0, 0, 0
            
            #age_majority_pred = torch.softmax(age_majority_pred,1)

            for b in age_binary_target:

                a = torch.argmax(b)
                pred = torch.argmax(majority_vote[index, :])

                if (a == pred):
                    age_correct += 1
                index += 1

            total += age_correct  # + gender_correct + race_correct

            nIm += dim
            # print("Gender accuracy: ",gender_correct/dim)
            #print("Age accuracy: ")
            #print(str(age_correct / dim))
            # print("Race accuracy: ",race_correct/dim)
            # print("Accuracy per batch size: ",(age_correct+gender_correct+race_correct)/(3*dim))
            iteration += 1
        print("Test")        
        print("Average accuracy: ")
        print(str(total / (nIm)))
            


def train_selection(selection_net, age_model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched):

    #n_models = len(age_model) + len(gender_model) + len(race_model)  # how many models in the overall / initial ensemble

    for m in age_model:
        m.eval()

    # This will predict the activations used to make a model selection

    C = args.c # how many models should be selected among n_models

    # simple unweighted knapsack solver, chooses C items with the largest scores
    # output is 0-1 vector where the 1's indicate chosen items

    def batch_knapsack(scores): #64000*55
        indices = torch.topk(scores, C).indices # 64000*21
        choice = torch.zeros_like( scores )
        choice.scatter_(1,indices,torch.ones(indices.shape,device=device))
        return choice

    # A 'differentiable' knapsack solver
    # See Berthet et al. 'Learning with Differentiable Perturbed Optimizers'
    # The parameters to this function may need adjusting

    knapsack_layer = perturbations.perturbations.perturbed_special(batch_knapsack,
                                                 num_samples=1000,
                                                 sigma=0.1,
                                                 noise='normal',
                                                 batched=True,
                                                 device=device,
                                                 hard_fwd=True)

    error = 0
    total = 0
    #random_total = 0
    nIm = 0
    #total_correct_dump_pred = 0
    iteration = 1
    #total_correct_unselected_predictions = 0
    correct_array = []
    total_dumb_calls = 0
    dumb_calls_array = []

    best = 10000.0  # early stopping
    best_model = copy.deepcopy(selection_net)
    fails = 0
    flag = False
    patience = 20

    for e in range(args.epochs):
        train_loss = 0
        for sample in trainDataLoader:

            data, target = sample['X'], sample['Y']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # probably not perfect wrt tensor orientation
            dim = target.shape[0]
            #optimizer.zero_grad()
            age_predictions = torch.zeros(len(age_model), dim, 7).to(device)
            #gender_predictions = torch.zeros(len(age_model), dim, 2)
            #race_predictions = torch.zeros(len(age_model), dim, 5)


            #age_predictions = torch.stack( [ torch.softmax(m(data),1) for m in age_model] ) #n_models*batch_size*_classes   #majority_voter(model,data)
            age_predictions = torch.stack( [ torch.softmax(m(data),1) for m in age_model] )
            #age_predictions = torch.stack([m(data) for m in age_model])
            #gender_predictions = torch.stack( [m(data) for m in gender_model] )
            #race_predictions = torch.stack( [m(data) for m in race_model] )
            selection_vals = selection_net(data) #batch_size*n_models
            #selection_vals.requires_grad_()
            selection_vals = torch.nn.functional.normalize(selection_vals)
            #selection_vals = torch.nn.Relu(selection_vals)
            if (args.injection):
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - torch.topk(age_predictions, 2, 2).values[:, :,1])
                selections = knapsack_layer(selection_vals*diff.T)
            else:
                selections = knapsack_layer(selection_vals) #batch_size*n_models

            #selection_vals = torch.nn.functional.normalize(selection_vals) # before I applied L2 normalization - torch.nn.functional.normalize(selection_vals)
            if (args.weight_pred):
                diff = (torch.topk(age_predictions, 2, 2).values[:, :, 0] - torch.topk(age_predictions, 2, 2).values[:, : ,1]).repeat(7, 1, 1)
                diff = torch.permute(diff, (1, 2, 0))
                age_predictions = age_predictions * selections.repeat(7, 1, 1).T * diff #selection_vals.repeat(7,1 , 1).T # * diff
            else:
                age_predictions = age_predictions * selections.repeat(7, 1, 1).T


            majority_vote = torch.zeros(dim,7).to(device)

            if (args.apply_sum):
                majority_vote = torch.sum(age_predictions,0)
            else:
                majority_vote = torch.mean(age_predictions,0)


            age_binary_target = torch.zeros((dim,7))

            age_binary_target = age_binary_target.to(device)

            index = 0
            p = 0
            with torch.no_grad():
                for t in target:
                    age_binary_target[index,t.item()]=1
                    index += 1

            if (args.use_softmax):
                majority_pred = torch.zeros(dim,7).to(device)
                majority_pred = torch.softmax(majority_vote,1)
                loss = loss_fun(majority_pred,age_binary_target) #+ loss_fun(gender_majority_pred, gender_binary_target) +\
            else:
                loss = loss_fun(majority_vote, age_binary_target)
            #loss_fun(race_majority_pred, race_binary_target)
            #loss.requires_grad_()
            if (loss.item()<0.1):
                print(majority_pred[1,:])
                print(age_binary_target[1,:])
            train_loss += loss.item()
            #app=torch.argmax(majority_vote,1)
            #app=torch.softmax(majority_vote,1)

            index = 0
            age_correct, gender_correct, race_correct = 0 , 0 , 0

            app = age_binary_target.cpu().detach().numpy()
            app2 = majority_vote.cpu().detach().numpy()

            for i in range(dim):
                t_max_index = np.argmax(app[i,:])
                p_max_index = np.argmax(app2[i,:])
                if (t_max_index == p_max_index):
                    age_correct += 1

            #total += age_correct #+ gender_correct + race_correct

            #nIm += dim
            if (args.clip):
                nn.utils.clip_grad_value_(selection_net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            if (args.sched):
                sched.step()
            #print("Gender accuracy: ",gender_correct/dim)
            #print("Race accuracy: ",race_correct/dim)
            #print("Accuracy per batch size: ",(age_correct+gender_correct+race_correct)/(3*dim))
            iteration += 1
            if (iteration%100==1):
            
                print("Loss function value: ")
                print(str(loss.item()))
            

        selection_net.eval()
        valid_loss = 0
        total = 0
        for sample in validDataLoader:

            d, t = sample['X'], sample['Y']
            if torch.cuda.is_available():
                d, t = d.cuda(), t.cuda()
            # probably not perfect wrt tensor orientation
            dim = t.shape[0]
            
            pred = torch.zeros(len(age_model), dim, 7).to(device)
            # gender_predictions = torch.zeros(len(age_model), dim, 2)
            # race_predictions = torch.zeros(len(age_model), dim, 5)

            pred = torch.stack([torch.softmax(m(d),1) for m in age_model])  # n_models*batch_size*_classes   #majority_voter(model,data)
            #age_predictions = torch.stack([m(data) for m in age_model])
            # gender_predictions = torch.stack( [m(data) for m in gender_model] )
            # race_predictions = torch.stack( [m(data) for m in race_model] )
            sel_vals = selection_net(d)  # batch_size*n_models
            # selection_vals.requires_grad_()
            sel_vals = torch.nn.functional.normalize(sel_vals)
            # selection_vals = torch.nn.Relu(selection_vals)
            if (args.injection):
                diff = (torch.topk(pred, 2, 2).values[:, :, 0] - torch.topk(pred, 2, 2).values[:, :,1])
                sel = knapsack_layer(sel_vals*diff.T)
            else:
                sel = knapsack_layer(sel_vals)  # batch_size*n_models

            # selection_vals = torch.nn.functional.normalize(selection_vals) # before I applied L2 normalization - torch.nn.functional.normalize(selection_vals)
            if (args.weight_pred):
                diff = (torch.topk(pred, 2, 2).values[:, :, 0] - torch.topk(pred, 2, 2).values[:, :, 1]).repeat(7, 1, 1)
                diff = torch.permute(diff, (1, 2, 0))
                pred = pred * sel.repeat(7, 1, 1).T * diff #selection_vals.repeat(7,1,1).T # diff
            else:
                pred = pred * sel.repeat(7, 1, 1).T

            maj_vote = torch.zeros(dim, 7).to(device)

            if (args.apply_sum):
                maj_vote = torch.sum(pred, 0)
            else:
                maj_vote = torch.mean(pred, 0)

            binary_target = torch.zeros((dim, 7))

            binary_target = binary_target.to(device)

            index = 0
            p = 0
            with torch.no_grad():
                for tt in t:
                    binary_target[index, tt.item()] = 1
                    index += 1

            if (args.use_softmax):
                maj_pred = torch.zeros(dim,7).to(device)
                maj_pred = torch.softmax(maj_vote, 1)
                loss = loss_fun(maj_pred,
                                binary_target)  # + loss_fun(gender_majority_pred, gender_binary_target) +\
            else:
                loss = loss_fun(maj_vote, binary_target)
            # loss_fun(race_majority_pred, race_binary_target)
            # loss.requires_grad_()

            print("Loss function value: ")
            print(str(loss.item()))

            # app=torch.argmax(majority_vote,1)
            # app=torch.softmax(majority_vote,1)

            index = 0
            correct = 0

            app = binary_target.cpu().detach().numpy()
            app2 = maj_pred.cpu().detach().numpy()

            for i in range(dim):
                t_max_index = np.argmax(app[i, :])
                p_max_index = np.argmax(app2[i, :])
                if (t_max_index == p_max_index):
                    correct += 1

            total += correct  # + gender_correct + race_correct

            nIm += dim
            # print("Gender accuracy: ",gender_correct/dim)
            # print("Race accuracy: ",race_correct/dim)
            # print("Accuracy per batch size: ",(age_correct+gender_correct+race_correct)/(3*dim))
            iteration += 1

            valid_loss += loss.item()

            # calculate average losses
        print("Epoch: ",e)
        print("Average accuracy: ")
        print(str(total / (nIm)))
        train_loss = train_loss / len(trainDataLoader.sampler)
        valid_loss = valid_loss / len(validDataLoader.sampler)

        print('Training Loss: {:.6f} \tValidation Loss: {:.6f}', train_loss, valid_loss)

        if valid_loss < (best - 1e-4):
            best_model = copy.deepcopy(selection_net)
            torch.save(best_model.state_dict(),"best_model_"+str(args.c))
            fails = 0
            best = valid_loss
        else:
            fails = fails + 1
        if fails > patience:
            print("Early Stopping. Valid hasn't improved for {}".format(patience))
            flag = True
        if flag:
            break

def main():
# Training settings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--c', type=int, default=5, metavar='N',
                        help='number of models of the ensemble')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--primary_class',type=list,default=[0],
                        help='class(es) the model is mainly trained on')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model_name', default=None,
                        help='Model name for saving - will not save model if name is None')
    parser.add_argument('--use_softmax', type=bool, default=True,
                        help='Apply softmax to 1/2 class predictions')
    parser.add_argument('--weight_pred', type=bool, default=False,
                        help='Weight predictions by model confidence')
    parser.add_argument('--injection', type=bool, default=False,
                        help='Inject knowledge to kp')
    parser.add_argument('--apply_sum', type=bool, default=True,
                        help='Apply sum to the 1/2 class soft predictions')
    parser.add_argument('--use_cvxpy', type=bool, default=False,
                        help='Use cvxpy library to solve the knapsack problem')
    parser.add_argument('--use_resnet', type=bool, default=False,
                        help='Use resnet for selecting base learners')
    parser.add_argument('--softmax_temperature',type=int, default=1, metavar='N',
                        help='softmax temperature')
    parser.add_argument('--train_only_last_layer',type=bool, default=True,
                        help='If false we train only the last layer of the network')
    parser.add_argument('--clip',type=bool, default=False,
                        help='If false we do not clip the gradient')
    parser.add_argument('--sched',type=bool, default=False,
                        help='If false we do not use the scheduler')

    args = parser.parse_args()
    
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    
    training_data = []
    training_label = []
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

    for batch_sample in train_ds:

        data = batch_sample[0].numpy()
        label = batch_sample[1]

        training_data.append(data)
        training_label.append(label)

    X = np.squeeze(training_data)
    Y = np.asarray(training_label) #Y: (dataset_size, 3)
    # Parameters

    # Random shuffle data
    X, Y = shuffle(X, Y)

    # torch.manual_seed(1)
    # Train-Test-Validation split
    train_valid_data = np.array((X[:int(len(X)*4/5)]))
    train_valid_labels = np.array((Y[:int(len(X)*4/5)]))
    #train_valid_data = np.array(X)
    #train_valid_labels = np.array(Y)
    test_data = np.array((X[int(len(X)*4/5)+1:]))
    test_labels = np.array((Y[int(len(X)*4/5)+1:]))
    #test_data = np.array(test_X)
    #test_labels = np.array(test_Y)

    train_data, valid_data, train_labels, valid_labels = train_test_split(train_valid_data, train_valid_labels,test_size=0.2)
    print(len(train_data))
    print(len(valid_data))
    trainSignData = D(train_valid_data, train_valid_labels,"rgb")
    trainDataLoader = torch.utils.data.DataLoader(trainSignData, shuffle=True, batch_size=args.batch_size)

    testSignData = D(test_data, test_labels,"rgb")
    testDataLoader = torch.utils.data.DataLoader(testSignData, shuffle=True, batch_size=args.batch_size)

    validSignData = D(valid_data, valid_labels,"rgb")
    validDataLoader = torch.utils.data.DataLoader(validSignData, shuffle=True, batch_size=args.batch_size)
    
    n_classes = 7
    #Net = torchvision.models.resnet18()
    #Net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model= torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULTO)
    Net = ResNet9(3, n_classes).to(device)
    #num_ftrs = Net.fc.in_features
    #Net.fc = nn.Linear(num_ftrs, 5)


    model =     [copy.deepcopy(Net), copy.deepcopy(Net), copy.deepcopy(Net),
                 copy.deepcopy(Net), copy.deepcopy(Net), copy.deepcopy(Net),
                 copy.deepcopy(Net), copy.deepcopy(Net), copy.deepcopy(Net),
                 copy.deepcopy(Net), copy.deepcopy(Net), copy.deepcopy(Net),
                 copy.deepcopy(Net), copy.deepcopy(Net), copy.deepcopy(Net),
                 copy.deepcopy(Net), copy.deepcopy(Net), copy.deepcopy(Net),                    copy.deepcopy(Net), copy.deepcopy(Net), copy.deepcopy(Net),
                 copy.deepcopy(Net), copy.deepcopy(Net), copy.deepcopy(Net),
                 copy.deepcopy(Net), copy.deepcopy(Net), copy.deepcopy(Net),
                 copy.deepcopy(Net)]

    path = 'spec_models'

    model[0].load_state_dict(torch.load(path+'/model_0', map_location=torch.device('cpu')))
    model[1].load_state_dict(torch.load(path+'/model_1',map_location=torch.device('cpu')))
    model[2].load_state_dict(torch.load(path+'/model_2',map_location=torch.device('cpu')))
    model[3].load_state_dict(torch.load(path+'/model_3',map_location=torch.device('cpu')))
    model[4].load_state_dict(torch.load(path+'/model_4',map_location=torch.device('cpu')))
    model[5].load_state_dict(torch.load(path+'/model_5',map_location=torch.device('cpu')))
    model[6].load_state_dict(torch.load(path+'/model_6',map_location=torch.device('cpu')))
    model[7].load_state_dict(torch.load(path+'/model_01',map_location=torch.device('cpu')))
    model[8].load_state_dict(torch.load(path+'/model_02',map_location=torch.device('cpu')))
    model[9].load_state_dict(torch.load(path+'/model_03',map_location=torch.device('cpu')))
    model[10].load_state_dict(torch.load(path+'/model_04',map_location=torch.device('cpu')))
    model[11].load_state_dict(torch.load(path+'/model_05',map_location=torch.device('cpu')))
    model[12].load_state_dict(torch.load(path+'/model_06',map_location=torch.device('cpu')))
    model[13].load_state_dict(torch.load(path+'/model_12',map_location=torch.device('cpu')))
    model[14].load_state_dict(torch.load(path+'/model_13',map_location=torch.device('cpu')))
    model[15].load_state_dict(torch.load(path+'/model_14',map_location=torch.device('cpu')))
    model[16].load_state_dict(torch.load(path+'/model_15',map_location=torch.device('cpu')))
    model[17].load_state_dict(torch.load(path+'/model_16',map_location=torch.device('cpu')))
    model[18].load_state_dict(torch.load(path+'/model_23',map_location=torch.device('cpu')))
    model[19].load_state_dict(torch.load(path+'/model_24',map_location=torch.device('cpu')))
    model[20].load_state_dict(torch.load(path+'/model_25',map_location=torch.device('cpu')))
    model[21].load_state_dict(torch.load(path+'/model_26',map_location=torch.device('cpu')))
    model[22].load_state_dict(torch.load(path+'/model_34',map_location=torch.device('cpu')))
    model[23].load_state_dict(torch.load(path+'/model_35',map_location=torch.device('cpu')))
    model[24].load_state_dict(torch.load(path+'/model_36',map_location=torch.device('cpu')))
    model[25].load_state_dict(torch.load(path+'/model_45',map_location=torch.device('cpu')))
    model[26].load_state_dict(torch.load(path+'/model_46',map_location=torch.device('cpu')))
    model[27].load_state_dict(torch.load(path+'/model_56',map_location=torch.device('cpu')))
    
    for i in range(len(model)):
        for param in model[i].parameters():
            param.requires_grad_(False)
        model[i].to(device)

    n_models = len(model) #+ len(gender_model) + len(race_model)  # how many models in the overall / initial ensemble
    if (args.use_resnet):
        selection_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
        if (args.train_only_last_layer):
            for param in selection_net.parameters():
                param.requires_grad = False
        num_ftrs = selection_net.fc.in_features
        selection_net.fc = nn.Linear(num_ftrs, n_models).to(device)
    else:
        selection_net = ResNet9(3, n_models).to(device)
    initialize_model = True
    param_distribution = 'xavier'
    if initialize_model:
        init_weights(selection_net, param_distribution) 
    selection_net.train()
    params = selection_net.parameters()
    loss_fun = torch.nn.CrossEntropyLoss()
    #optimizer = optim.Adadelta(params, lr=args.lr)
    #optimizer = torch.optim.Adam(params, weight_decay=0.001)
    optimizer = torch.optim.Adam(params, lr=0.0001)
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    max_lr = 0.001
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=args.epochs,
                                                steps_per_epoch=len(trainDataLoader))
    #for i in range(args.epochs):
    train_selection(selection_net, model, device, trainDataLoader, validDataLoader, optimizer, args, loss_fun, n_models, sched)
    best_selection_net = ResNet9(3,n_models).to(device)
    best_selection_net.load_state_dict(torch.load("best_model_"+str(args.c)))
    test(best_selection_net, model, device, testDataLoader,args)

if __name__ == '__main__':
    main()
