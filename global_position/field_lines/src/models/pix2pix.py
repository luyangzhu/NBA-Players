import torch
import torch.nn as nn

__all__ = ['weights_init', 'Pix2pixGen']

def weights_init(m):
    conv_tuple = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    batchnorm_tuple = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    if isinstance(m, conv_tuple):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, batchnorm_tuple):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.constant_(m.bias, 0)


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_nc)
        )
    def forward(self, inp):
        return self.block(inp)


class UpConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, isDropOut=False):
        super(UpConvBlock, self).__init__()
        self.upblock = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_nc)
        )
        self.dropout = nn.Dropout(p=0.5)
        self.isDropOut = isDropOut
    def forward(self, inp):
        out = self.upblock(inp)
        if self.isDropOut:
            out = self.dropout(out)
        return out


class Pix2pixGen(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, conv_chans=64):
        super(Pix2pixGen, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, conv_chans, kernel_size=4, stride=2, padding=1)
        self.conv_block1 = ConvBlock(conv_chans, conv_chans*2)
        self.conv_block2 = ConvBlock(conv_chans*2, conv_chans*4)
        self.conv_block3 = ConvBlock(conv_chans*4, conv_chans*8)
        self.conv_block4 = ConvBlock(conv_chans*8, conv_chans*8)
        self.conv_block5 = ConvBlock(conv_chans*8, conv_chans*8)
        self.conv_block6 = ConvBlock(conv_chans*8, conv_chans*8)
        self.conv_block7 = nn.Sequential(
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(conv_chans*8, conv_chans*8, kernel_size=4, stride=2, padding=1)
        )

        self.upconv_block1 = UpConvBlock(conv_chans*8, conv_chans*8, isDropOut=True)
        self.upconv_block2 = UpConvBlock(conv_chans*8*2, conv_chans*8, isDropOut=True)
        self.upconv_block3 = UpConvBlock(conv_chans*8*2, conv_chans*8, isDropOut=True)
        self.upconv_block4 = UpConvBlock(conv_chans*8*2, conv_chans*8)
        self.upconv_block5 = UpConvBlock(conv_chans*8*2, conv_chans*4)
        self.upconv_block6 = UpConvBlock(conv_chans*4*2, conv_chans*2)
        self.upconv_block7 = UpConvBlock(conv_chans*2*2, conv_chans)
        self.upconv_block8 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(conv_chans*2, output_nc, kernel_size=4, stride=2, padding=1)
        )
        self.tanh = nn.Tanh()

    def forward(self, inp):
        e1 = self.conv1(inp)
        e2 = self.conv_block1(e1)
        e3 = self.conv_block2(e2)
        e4 = self.conv_block3(e3)
        e5 = self.conv_block4(e4)
        e6 = self.conv_block5(e5)
        e7 = self.conv_block6(e6)
        e8 = self.conv_block7(e7)

        d1_ = self.upconv_block1(e8)
        d1 = torch.cat((d1_,e7),1)
        d2_ = self.upconv_block2(d1)
        d2 = torch.cat((d2_,e6),1)
        d3_ = self.upconv_block3(d2)
        d3 = torch.cat((d3_,e5),1)
        d4_ = self.upconv_block4(d3)
        d4 = torch.cat((d4_,e4),1)
        d5_ = self.upconv_block5(d4)
        d5 = torch.cat((d5_,e3),1)
        d6_ = self.upconv_block6(d5)
        d6 = torch.cat((d6_,e2),1)
        d7_ = self.upconv_block7(d6)
        d7 = torch.cat((d7_,e1),1)
        d8 = self.upconv_block8(d7)
        # out = self.tanh(d8)
        out = d8
        return out
