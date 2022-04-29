
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
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score
from matplotlib import pyplot as plt


#import a custom model
from model import CnnModel
import utils

classes = ['Non-Rooftop','Rooftop']
print('Instantiating the model..')
convNet = CnnModel()
print('Loading weights..')
convNet.load_state_dict(torch.load('./chkpoints/parameters'))
print('Making Predictions...')

def display(testloader,num_images):    
    
    x_test = iter(testloader)
    for k in range(num_images):
        (imgs,labels) = x_test.next() 
        outputs = convNet(imgs)
        #extra
        fig,ax = plt.subplots(1,1,figsize=(3,3))
         
        for img,label in zip(imgs,(outputs>0.5).float()):
          
              img = img/2 + 0.5              
              npimg = img.numpy()        
              ax.imshow(np.transpose(npimg, (1, 2, 0)))
              ax.set_xlabel('Prediction: '+classes[label.int()])
              ax.set_xticks([                                                                                                                                                                                                                                                ])
              ax.set_yticks([])

        plt.show()
        
        
      

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evalaute the model')
    
    parser.add_argument('num_images',
                        metavar = 'N',
                        type = int,
                        help = 'numebr of images to test')
    
    args = parser.parse_args()    
    num_images = args.num_images
    print('Loading test data.....')
    data = utils.load_data()
    testloader = DataLoader(data['test'], batch_size=4,shuffle=True)
    print('It will display predictions for {0} images one by one selected randomly from the test dataset'.format(num_images))
    
    display(testloader,num_images)

    

