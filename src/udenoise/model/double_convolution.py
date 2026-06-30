import torch.nn as nn


class DoubleConvolution(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_2d_1 = nn.Conv2d(
            self.input_channels, self.output_channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_2d_2 = nn.Conv2d(
            self.output_channels, self.output_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.conv_2d_1.forward(x)
        x = self.relu.forward(x)
        x = self.conv_2d_2.forward(x)
        return x
