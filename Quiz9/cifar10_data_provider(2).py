import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np

def download_data():
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

  test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)

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
  imshow(torchvision.utils.make_grid(images))
  # print labels
  print(' '.join('%5s' % classes[labels[j]] for j in range(4)))