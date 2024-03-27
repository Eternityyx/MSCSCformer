import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

def kaiming_init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.0)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activate="lrelu"):
        super(Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activate = activate

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

        # self.batch = nn.BatchNorm2d(self.out_channels)

        if activate == "lrelu":
            self.act = nn.LeakyReLU(0.2)
        elif activate == "tanh":
            self.act = nn.Tanh()
        elif activate == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activate == 'relu':
            self.act = nn.ReLU()
        elif activate == 'prelu':
            self.act = nn.PReLU()
        else:
            self.act = None

        layers = filter(lambda x: x is not None, [self.conv, self.act])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        return x
