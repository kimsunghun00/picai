import torch
import torch.nn as nn
from monai.networks.nets import UNet


class PiCaiNet(nn.Module):
    def __init__(self, model_depth=32):
        super().__init__()
        self.t2w_pass = UNet(spatial_dims=3,
                             in_channels=1,
                             out_channels=1,
                             channels=(16, 32, 64),
                             strides=(2, 2),
                             num_res_units=2)
        self.dwi_pass = UNet(spatial_dims=3,
                             in_channels=1,
                             out_channels=1,
                             channels=(16, 32, 64),
                             strides=(2, 2),
                             num_res_units=2)
        self.adc_pass = UNet(spatial_dims=3,
                             in_channels=1,
                             out_channels=1,
                             channels=(16, 32, 64),
                             strides=(2, 2),
                             num_res_units=2)

        self.conv = nn.Sequential(nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(),
                                  nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(),
                                  nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(),
                                  nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0),
                                  nn.LeakyReLU(),
                                  nn.Conv3d(32, 16, kernel_size=1, stride=1, padding=0),
                                  nn.LeakyReLU(),
                                  nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0)
                                  )

    def forward(self, images):
        t2w_output = torch.sigmoid(self.t2w_pass(images[:, 0, :, :, :].unsqueeze(dim=1)))
        dwi_output = torch.sigmoid(self.dwi_pass(images[:, 1, :, :, :].unsqueeze(dim=1)))
        adc_output = torch.sigmoid(self.adc_pass(images[:, 2, :, :, :].unsqueeze(dim=1)))

        stacked_output = torch.cat([t2w_output, dwi_output, adc_output], dim=1)

        out = torch.sigmoid(self.conv(stacked_output))

        return out.squeeze(), t2w_output.squeeze(), dwi_output.squeeze(), adc_output.squeeze()










class GreenBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, num_groups=4):
        super().__init__()
        self.res = nn.Conv3d(channel_in, channel_out, kernel_size=1, stride=1, padding=0)
        self.block = nn.Sequential(nn.GroupNorm(num_groups, channel_in),
                                   nn.ReLU(),
                                   nn.Conv3d(channel_in, channel_out, kernel_size, stride=1, padding=1),
                                   nn.GroupNorm(num_groups, channel_out),
                                   nn.ReLU(),
                                   nn.Conv3d(channel_out, channel_out, kernel_size, stride=1, padding=1))

    def forward(self, x):
        inp_res = self.res(x)
        x = self.block(x)
        out = x + inp_res
        return out


class Down(nn.Module):
    def __init__(self, num_channel):
        super().__init__()
        self.down = nn.Conv3d(num_channel, num_channel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv3d = nn.Conv3d(channel_in, channel_out, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.upsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, model_depth=32, p=0.2):
        super().__init__()
        self.initial_conv = nn.Conv3d(3, model_depth, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout3d(p)

        self.green_block0 = GreenBlock(model_depth, model_depth)
        self.dropout0 = nn.Dropout3d(p)
        self.downsize0 = Down(model_depth)

        self.green_block10 = GreenBlock(model_depth, model_depth * 2)
        self.dropout10 = nn.Dropout3d(p)
        self.green_block11 = GreenBlock(model_depth * 2, model_depth * 2)
        self.dropout11 = nn.Dropout3d(p)
        self.downsize1 = Down(model_depth * 2)

        self.green_block20 = GreenBlock(model_depth * 2, model_depth * 4)
        self.dropout20 = nn.Dropout3d(p)
        self.green_block21 = GreenBlock(model_depth * 4, model_depth * 4)
        self.dropout21 = nn.Dropout3d(p)
        self.green_block22 = GreenBlock(model_depth * 4, model_depth * 4)

    def forward(self, x):
        # Blue Block x1
        x = self.initial_conv(x)
        x = self.dropout(x)

        # Green Block x1
        x1 = self.green_block0(x)
        x = self.dropout0(x1)
        x = self.downsize0(x)

        # Green Blocks x2
        x = self.green_block10(x)
        x = self.dropout10(x)
        x2 = self.green_block11(x)
        x = self.dropout11(x2)
        x = self.downsize1(x)

        # Green Blocks x3
        x = self.green_block20(x)
        x = self.dropout20(x)
        x = self.green_block21(x)
        x = self.dropout21(x)
        x3 = self.green_block22(x)

        return x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, model_depth=32):
        super().__init__()
        self.upsize0 = Up(model_depth * 4, model_depth * 2)
        self.green_block00 = GreenBlock(model_depth * 2, model_depth * 2)
        self.green_block01 = GreenBlock(model_depth * 2, model_depth * 2)

        self.upsize1 = Up(model_depth * 2, model_depth)
        self.green_block1 = GreenBlock(model_depth, model_depth)

        self.blue_block = nn.Conv3d(model_depth, model_depth, kernel_size=3, stride=1, padding=1)

        self.out_GT = nn.Conv3d(model_depth, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x3):
        # Green Block x2
        x = self.upsize0(x3) + x2
        x = self.green_block00(x)
        x = self.green_block01(x)

        # Green Block x 1
        x = self.upsize1(x) + x1
        x = self.green_block1(x)

        # Blue Block x1
        x = self.blue_block(x)

        # Output Block
        out_GT = torch.sigmoid(self.out_GT(x))

        return out_GT