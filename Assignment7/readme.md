# Modularized version of S7.

Entire code is divided into the following libraries with each containing different modules or functions
# 1. cifar10_data_provider.py

  download_data()
  Defines transforms and downloads the data with the transforms applied.
  
  get_train_test_loaders()
  Splits the data and returns train_loader and test_loader
  
  display()
  Displays sample data
