import torch
import torch.nn as nn

def nonlinearity(x):
    return x * torch.sigmoid(x)

class Nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nonlinearity(x)

def Normalize(in_channels, channels_per_group=8, dimension=None):
    num_groups = in_channels // channels_per_group
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def convert_1d_to_3d(x, block_resolution, channel_num):
    # shape of x: (batch_size, single_dense_length, embed_dim * token_type_num)
    batch_size = x.shape[0]
    x = x.permute(0, 2, 1).contiguous().view(batch_size, channel_num, block_resolution, block_resolution, block_resolution)
    return x

def convert_3d_to_1d(x, block_resolution, channel_num):
    # shape of x: (batch_size, channel_num, block_resolution, block_resolution, block_resolution)
    batch_size = x.shape[0]
    x = x.view(batch_size, channel_num, block_resolution*block_resolution*block_resolution)
    x = x.permute(0, 2, 1).contiguous().view(batch_size, block_resolution*block_resolution*block_resolution, channel_num)
    return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
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
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, dimension=3)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels, dimension=3)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
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

class ResUNet(nn.Module):
    def __init__(self, input_channels, block_resolution, ch_mult=[1, 2, 4], num_res_blocks=1, resamp_with_conv=True, dropout=0.2):
        super(ResUNet, self).__init__()

        self.input_channels = input_channels
        self.block_resolution = block_resolution
        self.dense_single_scene_len = block_resolution * block_resolution * block_resolution

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

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
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
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
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up) # prepend to get consistent order

    def forward(self, x, is_1d=True):
        # shape of x: (batch_size, single_dense_length, input_channels)
        if is_1d:
            # reshape
            try:
                x = convert_1d_to_3d(x, self.block_resolution, self.input_channels)
            except:
                import pdb; pdb.set_trace()
        
        # downsampling
        hs = [x]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1))
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if is_1d:
            # reshape
            h = convert_3d_to_1d(h, self.block_resolution, self.input_channels)

        return h
