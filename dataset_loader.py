import torchvision.transforms as transforms
from torchvision import datasets

#Define image transforms to be applied on all images
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Path to train and test data
train_directory = '~/data/train/'
test_directory = '~/data/test/'

def load_data():
  #load data from the path containing data folder

  data = {
      'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
      'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
  }

  return data







