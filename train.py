
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torch.utils.data import DataLoader

from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score

from matplotlib import pyplot as plt

#import a custom model
from model import CnnModel

import utils
import pdb
data = utils.load_data()
print(data['train'])
print(data['test'])
#constants
batch_size = 10
trainloader = DataLoader(data['train'], batch_size=batch_size,shuffle=True)
classes = ('non-rooftop','rooftop')
x_train = iter(trainloader)
imgs, labels = x_train.next()


#Instantiate the model
convNet = CnnModel()
Loss = nn.BCELoss()
optimizer = optim.SGD(convNet.parameters(), lr=0.001, momentum=0.9)

def train(epochs,to_save):
    train_loss = []
    train_accuracy = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        
        running_loss = 0.0
        running_acc = 0.0
            
        for i, (inputs,labels) in enumerate(trainloader):
                       
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = convNet(inputs)

           
            
            labels= torch.tensor(labels,dtype=torch.float)
            loss = Loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # Loss
            running_loss += loss.item()
            # Accuracy
            class_correct = (outputs>0.5).float()
            # ensure shape of target and labels are same
            class_correct = class_correct == torch.as_tensor(labels.reshape((len(labels),1)))
            
            running_acc += torch.sum(class_correct).item() # count_nonzero() didn't work, threfore workaround using max
         
            if (i+1)%16 == 0:
                
                total_samples = len(trainloader.dataset.targets)
                print('[%d, %5d] loss: %.3f Accuracy:%.3f' %
                (epoch + 1, i + 1, running_loss ,running_acc/total_samples)) # (i+1) for correct averaging over iteration  
                train_loss.append(running_loss)
                train_accuracy.append(running_acc/total_samples)
    if to_save:
        save(train_loss,train_accuracy)

    

def save(train_loss,train_accuracy):
    torch.save(convNet.state_dict(),'parameters')
    torch.save({'Loss':train_loss, 'accuracy':train_accuracy},'metrics')
    print('The model paramters and metrcis are saved at:', \
              os.path.dirname(os.path.realpath(__file__)))

##def count_parameters(convNet):
##    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Neural Network')
    parser.add_argument('epochs',
                        metavar = 'N',
                        type = int,
                        help = 'Number of epochs')
    parser.add_argument('save',metavar='s',
                        type = int,
                        help = '1 -- > Save parameters, 0 --> not save')
    args = parser.parse_args()
    
    epochs = args.epochs
    to_save = args.save
    train(epochs,to_save)
    
   
