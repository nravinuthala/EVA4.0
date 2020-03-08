# Modularized version of S7.

Entire code is divided into the following libraries with each containing different modules or functions
# 1. cifar10_data_provider.py
Contains methods to download the data, split them into train and test and provide loaders and also to display sample images.

# 2. my_trainer.py
Contains methods to to define train and test functions, run method which calls train and test methods for the specified epochs and display method to display the losses and accuracies.

# 3. my_model.py
Contains the model class definition which can be used to create the model object from the main script.

# 4. S7_Nagaraj_Work_modular.ipynb
Main script which loads the other libraries and call the methods to load data, define network, train, test and finally plot the loss and accuracy graphs.
