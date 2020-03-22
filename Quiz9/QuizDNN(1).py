import torch
import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block - x1 - 3, 32, 32
        #x2 = conv(x1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 3, 32, 32 output_size 16, 32, 32 rf 3
        
        #C1
        #x3 = conv(x1 + x2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=19, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 19, 32, 32 output_size 32, 32, 32 rf 5
        
        #T1
        #x4 = maxpool (x1 + x2 + x3)
        self.pool1 = nn.MaxPool2d(2, 2)
        # input size 51, 32, 32 output_size 51, 16, 16 rf 6
        #x5 = conv(x4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=51, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 51, 16, 16 output_size 16, 16, 16 rf 6

        #C2
        #x6 = conv(x4 + x5)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=67, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 67, 16, 16 output_size 32, 16, 16 rf 10
        #x7 = conv(x4 + x5 + x6)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=99, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 99, 16, 16 output_size 64, 16, 16 rf 14
        #x8 = pool (x5 + x6 + x7)
        self.pool2 = nn.MaxPool2d(2, 2)
        # input size 112, 16, 16 output_size 112, 8, 8 rf 16
        #x9 = conv(x8)
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=16, kernel_size=(1, 1), padding=0, dilation=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 112, 8, 8 output_size 16, 8, 8 rf 16

        #C3
        #x10 = conv(x8 + x9)
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 128, 8, 8 output_size 256, 8, 8 rf 24

        #T2
        
        #x11 = conv(x8 + x9 + x10)
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # input size 384, 8, 8 output_size 512, 8, 8 rf 32
        
        #Output block
        #GAP
        #x12 = gap(x11)
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )
        #input 512, 8, 8 output 512, 1, 1
        #x13 = fc(x12)
        #self.fc = nn.Linear(256*512, 1, bias=False)
        #self.fc = nn.Linear(64*4*1, 10, bias=False)
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )
        

    def forward(self, x1): # input x1 3 channels of size 32, 32
        #print("x1 shape :: ", x1.shape)
        x2 = self.conv1(x1)  #input x1 3, 32, 32, output x2 16, 32, 32
        #print("x2 shape :: ", x2.shape)
        x3 = self.conv2(torch.cat((x1, x2), dim=1)) #input x1+x2 19, 32, 32, output x3 32, 32, 32
        #print("x3 shape :: ", x3.shape)
        x4 = self.pool1(torch.cat((x1 , x2 , x3), dim=1)) #input 51, 32, 32, output 51, 16, 16
        #print("x4 shape :: ", x4.shape),
        x5 = self.conv3(x4) #input 51, 16, 16, output 16, 16, 16
        #print("x5 shape :: ", x5.shape)
        x6 = self.conv4(torch.cat((x4 , x5), dim=1)) #input 102, 16, 16, output 32, 16, 16 
        #print("x6 shape :: ", x6.shape)
        x7 = self.conv5(torch.cat((x4 , x5 , x6), dim=1)) #input 134, 16, 16, output 64, 16, 16 
        #print("x7 shape :: ", x7.shape)
        x8 = self.pool2(torch.cat((x5 , x6 , x7), dim=1)) 
        #print("x8 shape :: ", x8.shape)
        x9 = self.conv6(x8)
        #print("x9 shape :: ", x9.shape)
        x10 = self.conv7(torch.cat((x8 , x9), dim=1))
        #print("x10 shape :: ", x10.shape)
        x11 = self.conv8(torch.cat((x8 , x9 , x10), dim=1))
        #print("x11 shape :: ", x11.shape)
        x12 = self.gap(x11)
        #print("x12 shape :: ", x12.shape)
        #x13 = self.fc(x12)
        x13 = self.conv11(x12)
        #print("x13 shape :: ", x13.shape)
        
        x = x13.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)