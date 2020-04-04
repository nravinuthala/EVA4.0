import torch
from matplotlib import pyplot as plt
import numpy as np

channel_means = (0.49139968, 0.48215841, 0.44653091)
channel_stdevs = (0.24703223, 0.24348513, 0.26158784)

class NN_Analysis:

    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def find_misclassified(self, model, test_loader, device, num_imgs=25):
        images = []
        target_list = []
        predicted_list = []
        count = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                for d, t, p in zip(data, target, pred):
                    if t != p:
                        d = d.cpu().numpy()
                        t = t.cpu().numpy()
                        p = p.cpu().numpy()
                        #d = (d*0.3081)+0.1307 #denormalizing each image before inserting
                        d = (d*0.4734)+0.2009
                        images.append(d)
                        target_list.append(t)
                        predicted_list.append(p)
                        count += 1
                        if count == num_imgs:
                            return images, target_list, predicted_list
    def plot_misclassifieds(self, imgs, trgt_lst, pred_lst, num_imgs=25, save=False):
        class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

        fig = plt.figure(figsize=(20, ((num_imgs//5)+1)*3))
        for i in range(num_imgs):
            ax = fig.add_subplot((num_imgs//5)+1, 5, i+1)
            #ax.imshow(np.rollaxis(imgs[i], 0, 3).squeeze(), cmap='gray')
            ax.imshow(np.rollaxis(imgs[i], 0, 3).squeeze())
            ax.axis('off')
            #print('Actual : ' + str(trgt_lst[i]) + ' Predicted : ' + str(pred_lst[i]))
            actual = class_names[trgt_lst[i]]
            predicted = class_names[pred_lst[i][0]]
            ax.set_title('Actual : ' + actual + ' Predicted : ' + predicted)
        plt.show()