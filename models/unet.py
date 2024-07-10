import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.utils as utils
logger = utils.get_logger(__name__)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 using 3D convolutions"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv using 3D pool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv for 3D data"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # Adjust the in_channels for the DoubleConv after concatenation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # Note that in_channels for DoubleConv must account for concatenated channels
            self.conv = DoubleConv(in_channels // 2 + in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure the size matches for concatenation
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)  # Concatenation doubles the channel count
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, args, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up1 = Up(256, 128, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)  # Assuming 3D output
        if args.MODE == 'train':  # eval_iters only in test case
            self.criterion = FocalLoss(args)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def build_unet(n_channels, n_classes):
    torch.cuda.empty_cache()
    return UNet(n_channels=n_channels, n_classes=n_classes)

if __name__ == "__main__":
    model = build_unet(n_channels=1, n_classes=1)
    print(model)

