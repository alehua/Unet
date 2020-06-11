import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class U_net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
class Lite_U_net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(Lite_U_net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=6)
        self.Conv2 = conv_block(ch_in=6,ch_out=12)
        self.Conv3 = conv_block(ch_in=12,ch_out=24)
    
        self.Up3 = up_conv(ch_in=24,ch_out=12)
        self.Up_conv3 = conv_block(ch_in=24, ch_out=12)
        self.Up2 = up_conv(ch_in=12,ch_out=6)
        self.Up_conv2 = conv_block(ch_in=12, ch_out=6)

        self.Conv_1x1 = nn.Conv2d(6,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):                      # x  1*512*512
        # encoding path
        x1 = self.Conv1(x)                    # x1 6*512*512      

        x2 = self.Maxpool(x1)                 # x2 6*256*256
        x2 = self.Conv2(x2)                   # x2 12*256*256

        x3 = self.Maxpool(x2)                 # x3 12*128*128
        x3 = self.Conv3(x3)                   # x3 24*128*128
        # decoding + concat path

        d3 = self.Up3(x3)                     # d3 12*256*256
        d3 = torch.cat((x2,d3),dim=1)         # d2 24*256*256
        d3 = self.Up_conv3(d3)                # d3 12*256*256


        d2 = self.Up2(x2)                     # d2 6*512*512
        d2 = torch.cat((x1,d2),dim=1)         # d2 12*512*512
        d2 = self.Up_conv2(d2)                # d3 6*512*512

        d1 = self.Conv_1x1(d2)                # d1 1*512*512

        return d1