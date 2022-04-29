
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




#import a custom model
from model import CnnModel
import utils


def main(testloader):
    print('Instantiating the model..')
    convNet = CnnModel()
    print('Loading weights..')
    convNet.load_state_dict(torch.load('./chkpoints/parameters'))
    print('Making Predictions...')
    test_acc = 0.0
    total_sample = 0
    
    for i, (inputs,labels) in enumerate(testloader):
      outputs = convNet(inputs)
      labels= torch.tensor(labels,dtype=torch.float)
      predictions= (outputs>0.5).float()
      # ensure shape of target and labels are same
      class_correct = predictions == torch.as_tensor(labels.reshape((len(labels),1)))        
      total_sample += labels.shape[0]
      test_acc += torch.sum(class_correct).item()
   
    print('The test accuracy is: ',(test_acc/total_sample))
    
        


    

if __name__ == '__main__':
    
    print('Loading ground truth.....')
    data = utils.load_data()
    testloader = DataLoader(data['test'], batch_size=44,shuffle=False)
    
    main(testloader)

    

