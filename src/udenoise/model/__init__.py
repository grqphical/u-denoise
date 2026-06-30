import torch
import torch.nn as nn
from .double_convolution import DoubleConvolution


class UNet(nn.Module):
    """A U-Net neural network implemented in PyTorch"""

    def __init__(self, in_channels=4, out_channels=4) -> None:
        super().__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_convolution_1 = DoubleConvolution(in_channels, 64)
        self.down_convolution_2 = DoubleConvolution(64, 128)
        self.down_convolution_3 = DoubleConvolution(128, 256)
        self.down_convolution_4 = DoubleConvolution(256, 512)
        self.down_convolution_5 = DoubleConvolution(512, 1024)

        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up_convolution_1 = DoubleConvolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_convolution_2 = DoubleConvolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_convolution_3 = DoubleConvolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_convolution_4 = DoubleConvolution(128, 64)

        self.output = nn.Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)
        # *** DO NOT APPLY MAX POOL TO down_9 ***

        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))

        x = self.output(x)
        return x
