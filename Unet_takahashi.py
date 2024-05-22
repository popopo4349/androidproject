import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU() if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU() if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, relu=True, skip=True, SE1=False, SE2=False):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.skip = skip
        self.SE1 = SE1
        self.SE2 = SE2
        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if self.skip==True:
            if (in_chs == out_chs and self.stride == 1):
                self.shortcut = nn.Sequential()
            else:
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),)

        if self.SE1==True:
            self.se1 = SEBlock(mid_chs)
        if self.SE2==True:
            self.se2 = SEBlock(out_chs) 
        self.rl = nn.ReLU()


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)
        # x = self.rl(x)

        if self.SE1==True:
            x = self.se1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        if self.skip==True:
            x += self.shortcut(residual)

        x = self.rl(x)

        if self.SE2==True:
            x = self.se2(x)
        return x

class hard_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SEBlock(nn.Module):
    def __init__(self, channels, se_ratio=2):
        super().__init__()
        self.sigmoid = nn.Sigmoid()   
        # self.sigmoid = hard_sigmoid() 
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        reduction = channels//se_ratio
        if reduction == 0:
            reduction = 1 
        self.fc1 = nn.Linear(channels, reduction)
        self.fc2 = nn.Linear(reduction, channels)
        self.rl = nn.ReLU()

    def forward(self, x):    
        out = x
        out = self.pool(x)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.rl(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.shape[0], out.shape[1], 1,1).expand_as(x)
        return x*out

class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, SE1=False, SE2=False, CBAM1=False, CBAM2=False, CO1=False, CO2=False):
        super().__init__()
        self.SE1 = SE1
        self.SE2 = SE2
        self.CBAM1 = CBAM1
        self.CBAM2 = CBAM2
        self.CO1 = CO1
        self.CO2 = CO2

        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size = 3, padding="same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size = 3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.SE1==True:
            self.se1 = SEBlock(middle_channels)
        if self.CBAM1==True:
            self.ca1 = ChannelAttention(middle_channels)
            self.sp1 = SpatialAttention()
        if self.CO1==True:
            self.co1 = CoordAtt(middle_channels,middle_channels)  
        if self.SE2==True:
            self.se2 = SEBlock(out_channels)
        if self.CBAM2==True:
            self.ca2 = ChannelAttention(out_channels)
            self.sp2 = SpatialAttention()
        if self.CO2==True:
            self.co2 = CoordAtt(out_channels,out_channels)     

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        if self.SE1==True:
            x = self.se1(x)
            print(x.shape)
        if self.CBAM1==True:
            x = self.ca1(x)
            x = self.sp1(x)
        if self.CO1==True:
            x = self.co1(x)    

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        if self.SE2==True:
            x = self.se2(x) 
        if self.CBAM2==True:
            x = self.ca2(x)
            x = self.sp2(x)  
        if self.CO2==True:
            x = self.co2(x)              
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class dense_layer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(dense_layer, self).__init__()        
        self.conv1 = nn.Conv2d(in_channels, out_channels=growth_rate*4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(growth_rate*4, out_channels=growth_rate, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(growth_rate*4)
        self.rl = nn.ReLU()

    def forward(self, x):
        out = self.bn1(x)
        out = self.rl(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.rl(out)
        out = self.conv2(out)
        return out

class dence_Block(nn.Module):
    def __init__(self, in_channels,  out_channels):
        super(dence_Block, self).__init__()
        plus_channels = in_channels//4 
        if plus_channels == 0:
            plus_channels = 1 
        self.dense1 = dense_layer(in_channels, plus_channels)
        self.dense2 = dense_layer(in_channels+plus_channels, plus_channels)
        self.dense3 = dense_layer(in_channels+2*plus_channels, plus_channels)
        self.dense4 = dense_layer(in_channels+3*plus_channels, plus_channels)
        self.conv = nn.Conv2d(in_channels=in_channels+plus_channels*4, out_channels=out_channels, kernel_size=1, stride=1,padding=0)
        self.bn = nn.BatchNorm2d(in_channels+plus_channels*4)
        self.rl = nn.ReLU()

    def forward(self,y):    
        y1 = self.dense1(y)
        y2 = torch.cat([y, y1], dim=1)
        y3 = self.dense2(y2)
        y4 = torch.cat([y2,y3], dim=1)
        y5 = self.dense3(y4)
        y6 = torch.cat([y4,y5], dim=1)
        y7 = self.dense4(y6)
        y8 = torch.cat([y6,y7], dim=1)
        y8 = self.bn(y8)
        y8 = self.rl(y8)
        y8 = self.conv(y8)
        return y8

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduction = in_planes//ratio
        if reduction == 0:
            reduction = 1           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, reduction, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(reduction, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return x*out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.bn(out)
        out = self.sigmoid(out)
        return x*out

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        # self.rl = nn.ReLU()
        self.rl = nn.ReLU6(inplace=True)
    
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        # y = self.rl(y)
        y = self.rl(y+3)/6 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class Ghost_skip(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = GhostBottleneck(3, 64, 64)
        self.TCB2 = GhostBottleneck(64, 128, 128)
        self.TCB3 = GhostBottleneck(128, 256, 256)
        self.TCB4 = GhostBottleneck(256, 512, 512)
        self.TCB5 = GhostBottleneck(512, 1024, 1024)
        self.TCB6 = GhostBottleneck(1024, 512, 512)
        self.TCB7 = GhostBottleneck(512, 256, 256)
        self.TCB8 = GhostBottleneck(256, 128, 128)
        self.TCB9 = GhostBottleneck(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4 = UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class Ghost_skip_SE_37(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = GhostBottleneck(3, 64, 64)
        self.TCB2 = GhostBottleneck(64, 128, 128)
        self.TCB3 = GhostBottleneck(128, 256, 256,SE1=True)
        self.TCB4 = GhostBottleneck(256, 512, 512)
        self.TCB5 = GhostBottleneck(512, 1024, 1024)
        self.TCB6 = GhostBottleneck(1024, 512, 512)
        self.TCB7 = GhostBottleneck(512, 256, 256,SE1=True)
        self.TCB8 = GhostBottleneck(256, 128, 128)
        self.TCB9 = GhostBottleneck(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4 = UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class Ghost(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = GhostBottleneck(3, 64, 64, skip=False)
        self.TCB2 = GhostBottleneck(64, 128, 128, skip=False)
        self.TCB3 = GhostBottleneck(128, 256, 256, skip=False)
        self.TCB4 = GhostBottleneck(256, 512, 512, skip=False)
        self.TCB5 = GhostBottleneck(512, 1024, 1024, skip=False)
        self.TCB6 = GhostBottleneck(1024, 512, 512, skip=False)
        self.TCB7 = GhostBottleneck(512, 256, 256, skip=False)
        self.TCB8 = GhostBottleneck(256, 128, 128, skip=False)
        self.TCB9 = GhostBottleneck(128, 64, 64, skip=False)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4 = UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64, SE1=True, SE2=True)
        self.TCB2 = TwoConvBlock(64, 128, 128, SE1=True, SE2=True)
        self.TCB3 = TwoConvBlock(128, 256, 256, SE1=True, SE2=True)
        self.TCB4 = TwoConvBlock(256, 512, 512, SE1=True, SE2=True)
        self.TCB5 = TwoConvBlock(512, 1024, 1024, SE1=True, SE2=True)
        self.TCB6 = TwoConvBlock(1024, 512, 512, SE1=True, SE2=True)
        self.TCB7 = TwoConvBlock(512, 256, 256, SE1=True, SE2=True)
        self.TCB8 = TwoConvBlock(256, 128, 128, SE1=True, SE2=True)
        self.TCB9 = TwoConvBlock(128, 64, 64, SE1=True, SE2=True)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE2(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 32, 32, SE1=True, SE2=True)
        self.TCB2 = TwoConvBlock(32, 64, 64, SE1=True, SE2=True)
        self.TCB3 = TwoConvBlock(64, 128, 128, SE1=True, SE2=True)
        self.TCB4 = TwoConvBlock(128, 256, 256, SE1=True, SE2=True)
        self.TCB5 = TwoConvBlock(256, 512, 512, SE1=True, SE2=True)
        self.TCB6 = TwoConvBlock(512, 256, 256, SE1=True, SE2=True)
        self.TCB7 = TwoConvBlock(256, 128, 128, SE1=True, SE2=True)
        self.TCB8 = TwoConvBlock(128, 64, 64, SE1=True, SE2=True)
        self.TCB9 = TwoConvBlock(64, 32, 32, SE1=True, SE2=True)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(512, 256) 
        self.UC2 = UpConv(256, 128) 
        self.UC3 = UpConv(128, 64) 
        self.UC4= UpConv(64, 32)

        self.conv1 = nn.Conv2d(32, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE3(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 16, 16, SE1=True, SE2=True)
        self.TCB2 = TwoConvBlock(16, 32, 32, SE1=True, SE2=True)
        self.TCB3 = TwoConvBlock(32, 64, 64, SE1=True, SE2=True)
        self.TCB4 = TwoConvBlock(64, 128, 128, SE1=True, SE2=True)
        self.TCB5 = TwoConvBlock(128, 256, 256, SE1=True, SE2=True)
        self.TCB6 = TwoConvBlock(256, 128, 128, SE1=True, SE2=True)
        self.TCB7 = TwoConvBlock(128, 64, 64, SE1=True, SE2=True)
        self.TCB8 = TwoConvBlock(64, 32, 32, SE1=True, SE2=True)
        self.TCB9 = TwoConvBlock(32, 16, 16, SE1=True, SE2=True)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE4(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 8, 8, SE1=True, SE2=True)
        self.TCB2 = TwoConvBlock(8, 16, 16, SE1=True, SE2=True)
        self.TCB3 = TwoConvBlock(16, 32, 32, SE1=True, SE2=True)
        self.TCB4 = TwoConvBlock(32, 64, 64, SE1=True, SE2=True)
        self.TCB5 = TwoConvBlock(64, 128, 128, SE1=True, SE2=True)
        self.TCB6 = TwoConvBlock(128, 64, 64, SE1=True, SE2=True)
        self.TCB7 = TwoConvBlock(64, 32, 32, SE1=True, SE2=True)
        self.TCB8 = TwoConvBlock(32, 16, 16, SE1=True, SE2=True)
        self.TCB9 = TwoConvBlock(16, 8, 8, SE1=True, SE2=True)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE3_1f9f(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 16, 16, SE1=True)
        self.TCB2 = TwoConvBlock(16, 32, 32)
        self.TCB3 = TwoConvBlock(32, 64, 64)
        self.TCB4 = TwoConvBlock(64, 128, 128)
        self.TCB5 = TwoConvBlock(128, 256, 256)
        self.TCB6 = TwoConvBlock(256, 128, 128)
        self.TCB7 = TwoConvBlock(128, 64, 64)
        self.TCB8 = TwoConvBlock(64, 32, 32)
        self.TCB9 = TwoConvBlock(32, 16, 16, SE1=True)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)


    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE_3f7f(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256, SE1=True)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256, SE1=True)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE_3b7b(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256, SE2=True)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256, SE2=True)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE3_3f7f(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 16, 16)
        self.TCB2 = TwoConvBlock(16, 32, 32)
        self.TCB3 = TwoConvBlock(32, 64, 64, SE1=True)
        self.TCB4 = TwoConvBlock(64, 128, 128)
        self.TCB5 = TwoConvBlock(128, 256, 256)
        self.TCB6 = TwoConvBlock(256, 128, 128)
        self.TCB7 = TwoConvBlock(128, 64, 64, SE1=True)
        self.TCB8 = TwoConvBlock(64, 32, 32)
        self.TCB9 = TwoConvBlock(32, 16, 16)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class CBAM_3f7f(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256, CBAM1=True)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256, CBAM1=True)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class CBAM3_3f7f(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 16, 16)
        self.TCB2 = TwoConvBlock(16, 32, 32)
        self.TCB3 = TwoConvBlock(32, 64, 64, CBAM1=True)
        self.TCB4 = TwoConvBlock(64, 128, 128)
        self.TCB5 = TwoConvBlock(128, 256, 256)
        self.TCB6 = TwoConvBlock(256, 128, 128)
        self.TCB7 = TwoConvBlock(128, 64, 64, CBAM1=True)
        self.TCB8 = TwoConvBlock(64, 32, 32)
        self.TCB9 = TwoConvBlock(32, 16, 16)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class CO3_3f7f(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 16, 16)
        self.TCB2 = TwoConvBlock(16, 32, 32)
        self.TCB3 = TwoConvBlock(32, 64, 64, CO1=True)
        self.TCB4 = TwoConvBlock(64, 128, 128)
        self.TCB5 = TwoConvBlock(128, 256, 256)
        self.TCB6 = TwoConvBlock(256, 128, 128)
        self.TCB7 = TwoConvBlock(128, 64, 64, CO1=True)
        self.TCB8 = TwoConvBlock(64, 32, 32)
        self.TCB9 = TwoConvBlock(32, 16, 16)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class CO3_f(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 16, 16, CO1=True)
        self.TCB2 = TwoConvBlock(16, 32, 32, CO1=True)
        self.TCB3 = TwoConvBlock(32, 64, 64, CO1=True)
        self.TCB4 = TwoConvBlock(64, 128, 128, CO1=True)
        self.TCB5 = TwoConvBlock(128, 256, 256, CO1=True)
        self.TCB6 = TwoConvBlock(256, 128, 128, CO1=True)
        self.TCB7 = TwoConvBlock(128, 64, 64, CO1=True)
        self.TCB8 = TwoConvBlock(64, 32, 32, CO1=True)
        self.TCB9 = TwoConvBlock(32, 16, 16, CO1=True)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class CO4_b(nn.Module):
    def __init__(self, senum=0):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 8, 8, CO2=True)
        self.TCB2 = TwoConvBlock(8, 16, 16, CO2=True)
        self.TCB3 = TwoConvBlock(16, 32, 32, CO2=True)
        self.TCB4 = TwoConvBlock(32, 64, 64, CO2=True)
        self.TCB5 = TwoConvBlock(64 ,128 ,128, CO2=True)
        self.TCB6 = TwoConvBlock(128, 64, 64, CO2=True)
        self.TCB7 = TwoConvBlock(64, 32, 32, CO2=True)
        self.TCB8 = TwoConvBlock(32, 16, 16, CO2=True)
        self.TCB9 = TwoConvBlock(16, 8, 8, CO2=True)  

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class CBAM3_f(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 16, 16, CBAM1=True)
        self.TCB2 = TwoConvBlock(16, 32, 32, CBAM1=True)
        self.TCB3 = TwoConvBlock(32, 64, 64, CBAM1=True)
        self.TCB4 = TwoConvBlock(64, 128, 128, CBAM1=True)
        self.TCB5 = TwoConvBlock(128, 256, 256, CBAM1=True)
        self.TCB6 = TwoConvBlock(256, 128, 128, CBAM1=True)
        self.TCB7 = TwoConvBlock(128, 64, 64, CBAM1=True)
        self.TCB8 = TwoConvBlock(64, 32, 32, CBAM1=True)
        self.TCB9 = TwoConvBlock(32, 16, 16, CBAM1=True)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class CO_f(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 64, 64, CO1=True)
        self.TCB2 = TwoConvBlock(64, 128, 128, CO1=True)
        self.TCB3 = TwoConvBlock(128, 256, 256, CO1=True)
        self.TCB4 = TwoConvBlock(256, 512, 512, CO1=True)
        self.TCB5 = TwoConvBlock(512, 1024, 1024, CO1=True)
        self.TCB6 = TwoConvBlock(1024, 512, 512, CO1=True)
        self.TCB7 = TwoConvBlock(512, 256, 256, CO1=True)
        self.TCB8 = TwoConvBlock(256, 128, 128, CO1=True)
        self.TCB9 = TwoConvBlock(128, 64, 64, CO1=True)
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class Dence(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = dence_Block(3, 64)
        self.TCB2 = dence_Block(64, 128)
        self.TCB3 = dence_Block(128, 256)
        self.TCB4 = dence_Block(256, 512)
        self.TCB5 = dence_Block(512, 1024)
        self.TCB6 = dence_Block(1024, 512)
        self.TCB7 = dence_Block(512, 256)
        self.TCB8 = dence_Block(256, 128)
        self.TCB9 = dence_Block(128, 64)
        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE3_selff(nn.Module):
    def __init__(self, senum=0):
        super().__init__()
        num = senum // 256
        muu = senum % 256 
        if num == 1:
            self.TCB1 = TwoConvBlock(3, 16, 16, SE1=True)
        else :
            self.TCB1 = TwoConvBlock(3, 16, 16)
        num = muu // 128
        muu = muu % 128 
        if num == 1:
            self.TCB2 = TwoConvBlock(16, 32, 32, SE1=True)
        else :
            self.TCB2 = TwoConvBlock(16, 32, 32)
        num = muu // 64
        muu = muu % 64 
        if num == 1:
            self.TCB3 = TwoConvBlock(32, 64, 64, SE1=True)
        else :
            self.TCB3 = TwoConvBlock(32, 64, 64)     
        num = muu // 32
        muu = muu % 32 
        if num == 1:
            self.TCB4 = TwoConvBlock(64, 128, 128, SE1=True)
        else :
            self.TCB4 = TwoConvBlock(64, 128, 128)     
        num = muu // 16
        muu = muu % 16 
        if num == 1:
            self.TCB5 = TwoConvBlock(128, 256, 256, SE1=True)
        else :
            self.TCB5 = TwoConvBlock(128, 256, 256)              
        num = muu // 8
        muu = muu % 8 
        if num == 1:
            self.TCB6 = TwoConvBlock(256, 128, 128, SE1=True)
        else :
            self.TCB6 = TwoConvBlock(256, 128, 128)   
        num = muu // 4
        muu = muu % 4 
        if num == 1:
            self.TCB7 = TwoConvBlock(128, 64, 64, SE1=True)
        else :
            self.TCB7 = TwoConvBlock(128, 64, 64) 
        num = muu // 2
        muu = muu % 2 
        if num == 1:
            self.TCB8 = TwoConvBlock(64, 32, 32, SE1=True)
        else :
            self.TCB8 = TwoConvBlock(64, 32, 32)       
        if muu == 1:
            self.TCB9 = TwoConvBlock(32, 16, 16, SE1=True)
        else :
            self.TCB9 = TwoConvBlock(32, 16, 16)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE3_selfb(nn.Module):
    def __init__(self, senum=0):
        super().__init__()
        num = senum // 256
        muu = senum % 256 
        if num == 1:
            self.TCB1 = TwoConvBlock(3, 16, 16, SE2=True)
        else :
            self.TCB1 = TwoConvBlock(3, 16, 16)
        num = muu // 128
        muu = muu % 128 
        if num == 1:
            self.TCB2 = TwoConvBlock(16, 32, 32, SE2=True)
        else :
            self.TCB2 = TwoConvBlock(16, 32, 32)
        num = muu // 64
        muu = muu % 64 
        if num == 1:
            self.TCB3 = TwoConvBlock(32, 64, 64, SE2=True)
        else :
            self.TCB3 = TwoConvBlock(32, 64, 64)     
        num = muu // 32
        muu = muu % 32 
        if num == 1:
            self.TCB4 = TwoConvBlock(64, 128, 128, SE2=True)
        else :
            self.TCB4 = TwoConvBlock(64, 128, 128)     
        num = muu // 16
        muu = muu % 16 
        if num == 1:
            self.TCB5 = TwoConvBlock(128, 256, 256, SE2=True)
        else :
            self.TCB5 = TwoConvBlock(128, 256, 256)              
        num = muu // 8
        muu = muu % 8 
        if num == 1:
            self.TCB6 = TwoConvBlock(256, 128, 128, SE2=True)
        else :
            self.TCB6 = TwoConvBlock(256, 128, 128)   
        num = muu // 4
        muu = muu % 4 
        if num == 1:
            self.TCB7 = TwoConvBlock(128, 64, 64, SE2=True)
        else :
            self.TCB7 = TwoConvBlock(128, 64, 64) 
        num = muu // 2
        muu = muu % 2 
        if num == 1:
            self.TCB8 = TwoConvBlock(64, 32, 32, SE2=True)
        else :
            self.TCB8 = TwoConvBlock(64, 32, 32)     
        if muu == 1:
            self.TCB9 = TwoConvBlock(32, 16, 16, SE2=True)
        else :
            self.TCB9 = TwoConvBlock(32, 16, 16)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(256, 128) 
        self.UC2 = UpConv(128, 64) 
        self.UC3 = UpConv(64, 32) 
        self.UC4= UpConv(32, 16)

        self.conv1 = nn.Conv2d(16, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE4_selff(nn.Module):
    def __init__(self, senum=0):
        super().__init__()
        num = senum // 256
        muu = senum % 256 
        if num == 1:
            self.TCB1 = TwoConvBlock(3, 8, 8, SE1=True)
        else :
            self.TCB1 = TwoConvBlock(3, 8, 8)
        num = muu // 128
        muu = muu % 128 
        if num == 1:
            self.TCB2 = TwoConvBlock(8, 16, 16, SE1=True)
        else :
            self.TCB2 = TwoConvBlock(8, 16, 16)
        num = muu // 64
        muu = muu % 64 
        if num == 1:
            self.TCB3 = TwoConvBlock(16, 32, 32, SE1=True)
        else :
            self.TCB3 = TwoConvBlock(16, 32, 32)     
        num = muu // 32
        muu = muu % 32 
        if num == 1:
            self.TCB4 = TwoConvBlock(32, 64, 64, SE1=True)
        else :
            self.TCB4 = TwoConvBlock(32, 64, 64)     
        num = muu // 16
        muu = muu % 16 
        if num == 1:
            self.TCB5 = TwoConvBlock(64 ,128 ,128, SE1=True)
        else :
            self.TCB5 = TwoConvBlock(64 ,128 ,128)              
        num = muu // 8
        muu = muu % 8 
        if num == 1:
            self.TCB6 = TwoConvBlock(128, 64, 64, SE1=True)
        else :
            self.TCB6 = TwoConvBlock(128, 64, 64)   
        num = muu // 4
        muu = muu % 4 
        if num == 1:
            self.TCB7 = TwoConvBlock(64, 32, 32, SE1=True)
        else :
            self.TCB7 = TwoConvBlock(64, 32, 32) 
        num = muu // 2
        muu = muu % 2 
        if num == 1:
            self.TCB8 = TwoConvBlock(32, 16, 16, SE1=True)
        else :
            self.TCB8 = TwoConvBlock(32, 16, 16)       
        if muu == 1:
            self.TCB9 = TwoConvBlock(16, 8, 8, SE1=True)
        else :
            self.TCB9 = TwoConvBlock(16, 8, 8)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE4_selfb(nn.Module):
    def __init__(self, senum=0):
        super().__init__()
        num = senum // 256
        muu = senum % 256 
        if num == 1:
            self.TCB1 = TwoConvBlock(3, 8, 8, SE2=True)
        else :
            self.TCB1 = TwoConvBlock(3, 8, 8)
        num = muu // 128
        muu = muu % 128 
        if num == 1:
            self.TCB2 = TwoConvBlock(8, 16, 16, SE2=True)
        else :
            self.TCB2 = TwoConvBlock(8, 16, 16)
        num = muu // 64
        muu = muu % 64 
        if num == 1:
            self.TCB3 = TwoConvBlock(16, 32, 32, SE2=True)
        else :
            self.TCB3 = TwoConvBlock(16, 32, 32)     
        num = muu // 32
        muu = muu % 32 
        if num == 1:
            self.TCB4 = TwoConvBlock(32, 64, 64, SE2=True)
        else :
            self.TCB4 = TwoConvBlock(32, 64, 64)     
        num = muu // 16
        muu = muu % 16 
        if num == 1:
            self.TCB5 = TwoConvBlock(64 ,128 ,128, SE2=True)
        else :
            self.TCB5 = TwoConvBlock(64 ,128 ,128)              
        num = muu // 8
        muu = muu % 8 
        if num == 1:
            self.TCB6 = TwoConvBlock(128, 64, 64, SE2=True)
        else :
            self.TCB6 = TwoConvBlock(128, 64, 64)   
        num = muu // 4
        muu = muu % 4 
        if num == 1:
            self.TCB7 = TwoConvBlock(64, 32, 32, SE2=True)
        else :
            self.TCB7 = TwoConvBlock(64, 32, 32) 
        num = muu // 2
        muu = muu % 2 
        if num == 1:
            self.TCB8 = TwoConvBlock(32, 16, 16, SE2=True)
        else :
            self.TCB8 = TwoConvBlock(32, 16, 16)       
        if muu == 1:
            self.TCB9 = TwoConvBlock(16, 8, 8, SE2=True)
        else :
            self.TCB9 = TwoConvBlock(16, 8, 8)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}



class CO4_selff(nn.Module):
    def __init__(self, senum=0):
        super().__init__()
        num = senum // 256
        muu = senum % 256 
        if num == 1:
            self.TCB1 = TwoConvBlock(3, 8, 8, CO1=True)
        else :
            self.TCB1 = TwoConvBlock(3, 8, 8)
        num = muu // 128
        muu = muu % 128 
        if num == 1:
            self.TCB2 = TwoConvBlock(8, 16, 16, CO1=True)
        else :
            self.TCB2 = TwoConvBlock(8, 16, 16)
        num = muu // 64
        muu = muu % 64 
        if num == 1:
            self.TCB3 = TwoConvBlock(16, 32, 32, CO1=True)
        else :
            self.TCB3 = TwoConvBlock(16, 32, 32)     
        num = muu // 32
        muu = muu % 32 
        if num == 1:
            self.TCB4 = TwoConvBlock(32, 64, 64, CO1=True)
        else :
            self.TCB4 = TwoConvBlock(32, 64, 64)     
        num = muu // 16
        muu = muu % 16 
        if num == 1:
            self.TCB5 = TwoConvBlock(64 ,128 ,128, CO1=True)
        else :
            self.TCB5 = TwoConvBlock(64 ,128 ,128)              
        num = muu // 8
        muu = muu % 8 
        if num == 1:
            self.TCB6 = TwoConvBlock(128, 64, 64, CO1=True)
        else :
            self.TCB6 = TwoConvBlock(128, 64, 64)   
        num = muu // 4
        muu = muu % 4 
        if num == 1:
            self.TCB7 = TwoConvBlock(64, 32, 32, CO1=True)
        else :
            self.TCB7 = TwoConvBlock(64, 32, 32) 
        num = muu // 2
        muu = muu % 2 
        if num == 1:
            self.TCB8 = TwoConvBlock(32, 16, 16, CO1=True)
        else :
            self.TCB8 = TwoConvBlock(32, 16, 16)       
        if muu == 1:
            self.TCB9 = TwoConvBlock(16, 8, 8, CO1=True)
        else :
            self.TCB9 = TwoConvBlock(16, 8, 8)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class CO4_selfb(nn.Module):
    def __init__(self, senum=0):
        super().__init__()
        num = senum // 256
        muu = senum % 256 
        if num == 1:
            self.TCB1 = TwoConvBlock(3, 8, 8, CO2=True)
        else :
            self.TCB1 = TwoConvBlock(3, 8, 8)
        num = muu // 128
        muu = muu % 128 
        if num == 1:
            self.TCB2 = TwoConvBlock(8, 16, 16, CO2=True)
        else :
            self.TCB2 = TwoConvBlock(8, 16, 16)
        num = muu // 64
        muu = muu % 64 
        if num == 1:
            self.TCB3 = TwoConvBlock(16, 32, 32, CO2=True)
        else :
            self.TCB3 = TwoConvBlock(16, 32, 32)     
        num = muu // 32
        muu = muu % 32 
        if num == 1:
            self.TCB4 = TwoConvBlock(32, 64, 64, CO2=True)
        else :
            self.TCB4 = TwoConvBlock(32, 64, 64)     
        num = muu // 16
        muu = muu % 16 
        if num == 1:
            self.TCB5 = TwoConvBlock(64 ,128 ,128, CO2=True)
        else :
            self.TCB5 = TwoConvBlock(64 ,128 ,128)              
        num = muu // 8
        muu = muu % 8 
        if num == 1:
            self.TCB6 = TwoConvBlock(128, 64, 64, CO2=True)
        else :
            self.TCB6 = TwoConvBlock(128, 64, 64)   
        num = muu // 4
        muu = muu % 4 
        if num == 1:
            self.TCB7 = TwoConvBlock(64, 32, 32, CO2=True)
        else :
            self.TCB7 = TwoConvBlock(64, 32, 32) 
        num = muu // 2
        muu = muu % 2 
        if num == 1:
            self.TCB8 = TwoConvBlock(32, 16, 16, CO2=True)
        else :
            self.TCB8 = TwoConvBlock(32, 16, 16)       
        if muu == 1:
            self.TCB9 = TwoConvBlock(16, 8, 8, CO2=True)
        else :
            self.TCB9 = TwoConvBlock(16, 8, 8)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class unet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 8, 8)
        self.TCB2 = TwoConvBlock(8, 16, 16)
        self.TCB3 = TwoConvBlock(16, 32, 32)     
        self.TCB4 = TwoConvBlock(32, 64, 64)     
        self.TCB5 = TwoConvBlock(64 ,128 ,128)              
        self.TCB6 = TwoConvBlock(128, 64, 64)   
        self.TCB7 = TwoConvBlock(64, 32, 32) 
        self.TCB8 = TwoConvBlock(32, 16, 16)       
        self.TCB9 = TwoConvBlock(16, 8, 8)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}


class Unet_self(nn.Module):
    def __init__(self, at='', in_chs=3, out_chs=64 , senum=0):
        super().__init__()
        num = senum // 256
        muu = senum % 256 
        if num == 1:
            self.TCB1 = TwoConvBlock(3, out_chs, out_chs, SE1=('SE1' in at), SE2=('SE2' in at), CBAM1=('CBAM1' in at), CBAM2=('CBAM2' in at), CO1=('CO1' in at), CO2=('CO2' in at))
        else :
            self.TCB1 = TwoConvBlock(3, out_chs, out_chs)
        num = muu // 128
        muu = muu % 128 
        if num == 1:
            self.TCB2 = TwoConvBlock(out_chs, out_chs*2, out_chs*2, SE1=('SE1' in at), SE2=('SE2' in at), CBAM1=('CBAM1' in at), CBAM2=('CBAM2' in at), CO1=('CO1' in at), CO2=('CO2' in at))
        else :
            self.TCB2 = TwoConvBlock(out_chs, out_chs*2, out_chs*2)
        num = muu // 64
        muu = muu % 64 
        if num == 1:
            self.TCB3 = TwoConvBlock(out_chs*2, out_chs*4, out_chs*4, SE1=('SE1' in at), SE2=('SE2' in at), CBAM1=('CBAM1' in at), CBAM2=('CBAM2' in at), CO1=('CO1' in at), CO2=('CO2' in at))
        else :
            self.TCB3 = TwoConvBlock(out_chs*2, out_chs*4, out_chs*4)     
        num = muu // 32
        muu = muu % 32 
        if num == 1:
            self.TCB4 = TwoConvBlock(out_chs*4, out_chs*8, out_chs*8, SE1=('SE1' in at), SE2=('SE2' in at), CBAM1=('CBAM1' in at), CBAM2=('CBAM2' in at), CO1=('CO1' in at), CO2=('CO2' in at))
        else :
            self.TCB4 = TwoConvBlock(out_chs*4, out_chs*8, out_chs*8)     
        num = muu // 16
        muu = muu % 16 
        if num == 1:
            self.TCB5 = TwoConvBlock(out_chs*8, out_chs*16, out_chs*16, SE1=('SE1' in at), SE2=('SE2' in at), CBAM1=('CBAM1' in at), CBAM2=('CBAM2' in at), CO1=('CO1' in at), CO2=('CO2' in at))
        else :
            self.TCB5 = TwoConvBlock(out_chs*8, out_chs*16, out_chs*16)              
        num = muu // 8
        muu = muu % 8 
        if num == 1:
            self.TCB6 = TwoConvBlock(out_chs*16, out_chs*8, out_chs*8, SE1=('SE1' in at), SE2=('SE2' in at), CBAM1=('CBAM1' in at), CBAM2=('CBAM2' in at), CO1=('CO1' in at), CO2=('CO2' in at))
        else :
            self.TCB6 = TwoConvBlock(out_chs*16, out_chs*8, out_chs*8)   
        num = muu // 4
        muu = muu % 4 
        if num == 1:
            self.TCB7 = TwoConvBlock(out_chs*8, out_chs*4, out_chs*4, SE1=('SE1' in at), SE2=('SE2' in at), CBAM1=('CBAM1' in at), CBAM2=('CBAM2' in at), CO1=('CO1' in at), CO2=('CO2' in at))
        else :
            self.TCB7 = TwoConvBlock(out_chs*8, out_chs*4, out_chs*4) 
        num = muu // 2
        muu = muu % 2 
        if num == 1:
            self.TCB8 = TwoConvBlock(out_chs*4, out_chs*2, out_chs*2, SE1=('SE1' in at), SE2=('SE2' in at), CBAM1=('CBAM1' in at), CBAM2=('CBAM2' in at), CO1=('CO1' in at), CO2=('CO2' in at))
        else :
            self.TCB8 = TwoConvBlock(out_chs*4, out_chs*2, out_chs*2)       
        if muu == 1:
            self.TCB9 = TwoConvBlock(out_chs*2, out_chs, out_chs, SE1=('SE1' in at), SE2=('SE2' in at), CBAM1=('CBAM1' in at), CBAM2=('CBAM2' in at), CO1=('CO1' in at), CO2=('CO2' in at))
        else :
            self.TCB9 = TwoConvBlock(out_chs*2, out_chs, out_chs)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(out_chs*16, out_chs*8) 
        self.UC2 = UpConv(out_chs*8, out_chs*4) 
        self.UC3 = UpConv(out_chs*4, out_chs*2) 
        self.UC4= UpConv(out_chs*2, out_chs)

        self.conv1 = nn.Conv2d(out_chs, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}



class SE4_f2567(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 8, 8)
        self.TCB2 = TwoConvBlock(8, 16, 16, SE1=True)
        self.TCB3 = TwoConvBlock(16, 32, 32)     
        self.TCB4 = TwoConvBlock(32, 64, 64)     
        self.TCB5 = TwoConvBlock(64 ,128 ,128, SE1=True)
        self.TCB6 = TwoConvBlock(128, 64, 64, SE1=True)
        self.TCB7 = TwoConvBlock(64, 32, 32, SE1=True)
        self.TCB8 = TwoConvBlock(32, 16, 16)       
        self.TCB9 = TwoConvBlock(16, 8, 8)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class SE4_b379(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 8, 8)
        self.TCB2 = TwoConvBlock(8, 16, 16)
        self.TCB3 = TwoConvBlock(16, 32, 32, SE2=True)     
        self.TCB4 = TwoConvBlock(32, 64, 64)     
        self.TCB5 = TwoConvBlock(64 ,128 ,128)
        self.TCB6 = TwoConvBlock(128, 64, 64)
        self.TCB7 = TwoConvBlock(64, 32, 32, SE2=True)
        self.TCB8 = TwoConvBlock(32, 16, 16)       
        self.TCB9 = TwoConvBlock(16, 8, 8,SE2=True)     

        self.maxpool = nn.MaxPool2d(2, stride = 2)

        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out': x}

class Unet_4_SEb(nn.Module):
    def __init__(self, senum=0):
        super().__init__()
        self.TCB1 = TwoConvBlock(3, 8, 8)
        self.TCB2 = TwoConvBlock(8, 16, 16)
        self.TCB3 = TwoConvBlock(16, 32, 32, SE2=True)
        self.TCB4 = TwoConvBlock(32, 64, 64)
        self.TCB5 = TwoConvBlock(64, 128, 128)
        self.TCB6 = TwoConvBlock(128, 64, 64)
        self.TCB7 = TwoConvBlock(64, 32, 32, SE2=True)
        self.TCB8 = TwoConvBlock(32, 16, 16)
        self.TCB9 = TwoConvBlock(16, 8, 8, SE2=True)
        self.maxpool = nn.MaxPool2d(2, stride = 2)    

        self.UC1 = UpConv(128, 64) 
        self.UC2 = UpConv(64, 32) 
        self.UC3 = UpConv(32, 16) 
        self.UC4= UpConv(16, 8)

        self.conv1 = nn.Conv2d(8, 3, kernel_size = 1)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return {'out':x, 'backbone_out':[x2, x3, x4]}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, SE1=False, CBAM1=False, CO1=False):
        super().__init__()
        self.SE1 = SE1
        self.CBAM1 = CBAM1
        self.CO1 = CO1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

        if self.SE1==True:
            self.se1 = SEBlock(out_channels)
        if self.CBAM1==True:
            self.ca1 = ChannelAttention(out_channels)
            self.sp1 = SpatialAttention()
        if self.CO1==True:
            self.co1 = CoordAtt(out_channels,out_channels)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        if self.SE1==True:
            x = self.se1(x)
            print(x.shape)
        if self.CBAM1==True:
            x = self.ca1(x)
            x = self.sp1(x)
        if self.CO1==True:
            x = self.co1(x)    
        return x

class InstanceSegmentation(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.conv1 = ConvBlock(16, 16, stride=2)
        self.conv2 = ConvBlock(32, 32)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(96, 32, stride=2)
        self.conv5 = ConvBlock(32, 5+classes)
        self.down1 = ConvBlock(64, 32)
        self.up1 = UpConv(16, 32)

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x1 = self.down(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x3 = self.up1(x3)
        x = torch.cat([x1, x2, x3], dim = 1)
        x = self.conv4(x)
        x = self.conv5(x)
        return x





