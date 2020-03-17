import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np

import cv2
from albumentations import Compose
from albumentations.pytorch import ToTensor
from albumentations import (VerticalFlip, RandomCrop, Normalize, HorizontalFlip, Flip, RandomRotate90, 
                            Rotate, Resize, ShiftScaleRotate, CenterCrop, OpticalDistortion, GridDistortion, 
                            ElasticTransform, JpegCompression, HueSaturationValue, RGBShift, RandomBrightness, 
                            RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CLAHE, ChannelShuffle, 
                            InvertImg, RandomGamma, ToGray, PadIfNeeded, Cutout, CoarseDropout)
class Alb_Transform_Train:
    def __init__(self):
        self.alb_transform = Compose([
            VerticalFlip(p=.5),
            HorizontalFlip(p=.5),
            CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=[0.4914*255, 0.4822*255, 0.4465*255], always_apply=False, p=0.5),
            HueSaturationValue(hue_shift_limit=(-25,0),sat_shift_limit=0,val_shift_limit=0,p=1),
            Rotate(p=.5, border_mode=cv2.BORDER_CONSTANT),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ToTensor()
        ])
    
    def __call__(self, img):
        img = np.array(img)
        img = self.alb_transform(image=img)['image']
        return img

class Alb_Transform_Test:
    def __init__(self):
        self.alb_transform = Compose([
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ToTensor()
        ])
    
    def __call__(self, img):
        img = np.array(img)
        img = self.alb_transform(image=img)['image']
        return img

def download_data():
  train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=Alb_Transform_Train())

  test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=Alb_Transform_Test())

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return train, test, classes

def get_train_test_loaders(train, test):
  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
    torch.cuda.manual_seed(SEED)

  # dataloader arguments - something you'll fetch these from cmdprmt
  trainloader_args = dict(shuffle=True, batch_size=256, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
  testloader_args = dict(shuffle=False, batch_size=100, num_workers=2, pin_memory=True) if cuda else dict(shuffle=False, batch_size=64)


  # train dataloader
  train_loader = torch.utils.data.DataLoader(train, **trainloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test, **testloader_args)

  return train_loader, test_loader

def display(train_loader, classes):
  # functions to show an image
  def imshow(img):
      img = img / 2 + 0.5     # unnormalize
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))

  # get some random training images
  dataiter = iter(train_loader)
  images, labels = dataiter.next()

  # show images
  #imshow(torchvision.utils.make_grid(images))
  imshow(images[1])
  print(classes[labels[1]])
  # print labels
  #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))