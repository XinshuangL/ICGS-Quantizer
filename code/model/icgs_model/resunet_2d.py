import torch
import torch.nn as nn

def nonlinearity(x): # swish
    return x*torch.sigmoid(x)

class Nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nonlinearity(x)

def Normalize(in_channels, channels_per_group=8, dimension=None):
    num_groups = in_channels // channels_per_group
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, dimension=2)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels, dimension=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class ResUNet2D(nn.Module):
    def __init__(self, input_channels, ch_mult=[1, 2, 4, 8, 16, 32], num_res_blocks=1, resamp_with_conv=True, dropout=0.2, downsample_order=3, use_dino=False, image_resolution=128, dino_resolution=16, dino_dim=384):
        super(ResUNet2D, self).__init__()

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.use_dino = use_dino
        self.input_channels = input_channels
        self.downsample_order = downsample_order
        self.upsample_order = self.num_resolutions - 1 - self.downsample_order

        self.image_resolution = image_resolution
        self.dino_resolution = dino_resolution

        print('self.upsample_order', self.upsample_order)

        cur_resolution = image_resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = input_channels*in_ch_mult[i_level]
            block_out = input_channels*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                cur_resolution = cur_resolution // 2
                if cur_resolution == dino_resolution and self.use_dino:
                    self.dino_merger = nn.Conv2d(block_in+dino_dim, block_in, 1)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleDict()
        for up_id, i_level in enumerate(range(self.num_resolutions - 1, self.downsample_order - 1, -1)):
            # print('up_id', up_id, 'i_level', i_level)

            block = nn.ModuleList()
            block_out = input_channels*ch_mult[i_level]
            skip_in = input_channels*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = input_channels*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != self.downsample_order:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up[str(up_id)] = up

    def forward(self, x, dino_features):
        cur_resolution = self.image_resolution

        # downsampling
        hs = [x] if self.downsample_order == 0 else []
        cur_h = x
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                cur_h = self.down[i_level].block[i_block](cur_h)
                if i_level >= self.downsample_order - 1: 
                    hs.append(cur_h)
            if i_level != self.num_resolutions-1:
                cur_h = self.down[i_level].downsample(cur_h)
                cur_resolution = cur_resolution // 2
                if cur_resolution == self.dino_resolution and self.use_dino:
                    cur_h = self.dino_merger(torch.cat([cur_h, dino_features], dim=1))
                if i_level >= self.downsample_order - 1: 
                    hs.append(cur_h)

        # middle
        h = hs[-1]
        h = self.mid.block(h)

        # upsampling
        # print('up max', self.num_resolutions - self.downsample_order)
        for up_id in range(self.num_resolutions - self.downsample_order):
            # print('forward up_id', up_id)
            for i_block in range(self.num_res_blocks+1):
                h = self.up[str(up_id)].block[i_block](
                    torch.cat([h, hs.pop()], dim=1))
            if up_id != self.num_resolutions - self.downsample_order - 1:
                # print('use upsample')
                h = self.up[str(up_id)].upsample(h)

        return h
