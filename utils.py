import torchvision.transforms as transforms
from torchvision import datasets

#Define image transforms to be applied on all images
image_transforms = {
    'train': transforms.Compose([        
        transforms.Resize(size=(64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(size=(64,64)),
         transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
}

# Path to train and test data
train_directory = './data/train/'
test_directory = './data/test/'

def load_data():
  #load data from the path containing data folder

  data = {
      'train': datasets.ImageFolder(root=train_directory,transform=image_transforms['train']),
      'test': datasets.ImageFolder(root=test_directory,transform=image_transforms['test'])
  }
  #
  #transform=image_transforms['test']
  return data







