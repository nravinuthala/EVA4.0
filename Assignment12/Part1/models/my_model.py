import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 32 output_size 32 rf 3
        
        #C1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 32 output_size 32 rf 5
        
        #T1
        self.pool1 = nn.MaxPool2d(2, 2)
        # input size 32 output_size 16 rf 6
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 16 output_size 16 rf 6

        #C2
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 16 output_size 16 rf 10
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 16 output_size 16 rf 14

        #T2
        self.pool2 = nn.MaxPool2d(2, 2)
        # input size 16 output_size 8 rf 16
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, dilation=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 8 output_size 8 rf 16

        #C3
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 8 output_size 8 rf 24
        #self.conv8 = nn.Sequential(
        #    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
        self.conv8_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), padding=1, groups=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 8 output_size 8 rf 32
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )# input size 8 output_size 8 rf 32
        #T3
        self.pool3 = nn.MaxPool2d(2, 2)
        # input size 8 output_size 4 rf 36
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 4 output_size 4 rf 36

        #C4
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 4 output_size 4 rf 52
    

        #Output block
        #GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )
        #self.fc = nn.Linear(64, 10, bias=False)
        #self.fc = nn.Linear(64*4*1, 10, bias=False)
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8_1(x)
        x = self.conv8_2(x)
        x = self.pool3(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        #x = self.fc(x)
        x = self.conv11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)