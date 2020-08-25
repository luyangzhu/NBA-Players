import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import models.resnet as resnet
import torch.nn.functional as F

#UTILITIES
class Bilinear(nn.Module):
    def __init__(self, linear_chans):
        super(Bilinear, self).__init__()
        self.linear_module1 = nn.Sequential(
            nn.Linear(linear_chans, linear_chans),
            nn.BatchNorm1d(linear_chans),
            nn.ReLU(),
            nn.Dropout()
        )
        self.linear_module2 = nn.Sequential(
            nn.Linear(linear_chans, linear_chans),
            nn.BatchNorm1d(linear_chans),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self,inp):
        hidden1 = self.linear_module1(inp)
        hidden2 = self.linear_module2(hidden1)
        out = hidden2 + inp
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        super(FeatureExtractor, self).__init__()
        self.encoder = resnet.resnet50(pretrained=True)
        self.last_lin = nn.Linear(2048, bottleneck_size)

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.last_lin(x)
        return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class IMG_AtlasNet_Humans(nn.Module):
    def __init__(self, num_points = 6036, bottleneck_size = 1024):
        super(IMG_AtlasNet_Humans, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = FeatureExtractor(bottleneck_size = self.bottleneck_size)
        self.decoder = PointGenCon(bottleneck_size = 3+self.bottleneck_size)


    def forward(self, x, template_vertices):
        x = self.encoder(x)
        rand_grid = template_vertices.transpose(1,2).contiguous()
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        deform_flow = self.decoder(y)
        out = deform_flow + rand_grid
        return out.contiguous().transpose(2,1).contiguous()
