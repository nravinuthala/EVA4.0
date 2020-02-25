S6 Assignment Notes:

1. Took base model and ran for 40 epochs.
2. Made a copy and added L1
3. Made a copy and added L2
4. Made a copy and added L1 and L2

After adding L1 and L2 and both together, the behavior of the model was strange. It was making it perform worse. Then I realized very late that, my base model was not overfitting at all. Instead it was slightly underfitting, hence adding regularizations were hindering the learning and reduced the train and test accuracies.

Could not complete the code to gather all accuracies and losses of the 4 models at once place to be able to plot them in one graph. Still working on it.

Could not complete the code to identify mis classified images for L1 and L2 models. Still working on it.

Will complete and resubmit.
