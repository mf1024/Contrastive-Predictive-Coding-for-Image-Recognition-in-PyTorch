from torch import nn

class ResNetBlock_v2(nn.Module):

    def __init__(self, in_channels_block, is_downsampling_block = False):
        super(ResNetBlock_v2, self).__init__()

        self.in_channels_block = in_channels_block
        self.out_channels_block = in_channels_block
        self.is_downsampling_block = is_downsampling_block

        self.layer_1_stride = 1

        if self.is_downsampling_block:
            self.out_channels_block *= 2
            self.layer_1_stride = 2

            self.projection_shortcut = nn.Conv2d(
                in_channels = self.in_channels_block,
                out_channels = self.out_channels_block,
                kernel_size = 1,
                stride = 2,
                padding = 0
            )

        self.batch_norm_1 = nn.BatchNorm2d(self.in_channels_block)
        self.relu_1 = nn.ReLU()
        self.conv_layer_1 = nn.Conv2d(
            in_channels = self.in_channels_block,
            out_channels = self.out_channels_block,
            kernel_size = 3,
            stride = self.layer_1_stride,
            padding = 1)

        self.batch_norm_2 = nn.BatchNorm2d(self.out_channels_block)
        self.relu_2 = nn.ReLU()
        self.conv_layer_2 = nn.Conv2d(
            in_channels = self.out_channels_block,
            out_channels = self.out_channels_block,
            kernel_size = 3,
            stride = 1,
            padding = 1)

    def forward(self,x):

        identity = x

        if self.is_downsampling_block:
            identity = self.projection_shortcut.forward(identity)

        x = self.batch_norm_1.forward(x)
        x = self.relu_1.forward(x)
        x = self.conv_layer_1.forward(x)

        x = self.batch_norm_2.forward(x)
        x = self.relu_2.forward(x)
        x = self.conv_layer_2.forward(x)

        x = x + identity

        return x


class SqueezeAndExcitationBlock(nn.Module):

    def __init__(self, r, channels):
        super(SqueezeAndExcitationBlock, self).__init__()

        self.bottleneck_features = channels//r

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=channels, out_features=self.bottleneck_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.bottleneck_features, out_features=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        conv = x

        x = self.avg_pool.forward(x)
        x = x.squeeze(dim=3)
        x = x.squeeze(dim=2)
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        x = self.sigmoid.forward(x)
        x = x.unsqueeze(dim=2)
        x = x.unsqueeze(dim=3)

        conv = conv * x

        return conv


class SE_ResNetBottleneckBlock(nn.Module):

    def __init__(self, in_channels_block, is_downsampling_block = False):
        super(SE_ResNetBottleneckBlock, self).__init__()

        self.is_downsampling_block = is_downsampling_block
        self.in_channels_block = in_channels_block
        self.bottleneck_channels = in_channels_block // 4
        self.out_channels_block = self.bottleneck_channels * 4
        self.layer_1_stride = 1


        if self.is_downsampling_block:
            self.bottleneck_channels *= 2
            self.out_channels_block *= 2
            self.layer_1_stride = 2

            self.projection_shortcut = nn.Conv2d(
                in_channels = self.in_channels_block,
                out_channels = self.out_channels_block,
                kernel_size = 1,
                stride = 2,
                padding = 0
            )


        self.squeeze_and_excitation = SqueezeAndExcitationBlock(
                    r = 16,
                    channels = self.out_channels_block
            )

        self.batch_norm_1 = nn.BatchNorm2d(self.in_channels_block)
        self.relu_1 = nn.ReLU()
        self.conv_layer_1 = nn.Conv2d(
            in_channels=self.in_channels_block,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            stride=self.layer_1_stride,
            padding=0)

        self.batch_norm_2 = nn.BatchNorm2d(self.bottleneck_channels)
        self.relu_2 = nn.ReLU()
        self.conv_layer_2 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.bottleneck_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        self.batch_norm_3 = nn.BatchNorm2d(self.bottleneck_channels)
        self.relu_3 = nn.ReLU()
        self.conv_layer_3 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.out_channels_block,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

    def forward(self,x):

        identity = x

        if self.is_downsampling_block:
            identity = self.projection_shortcut.forward(identity)

        x = self.batch_norm_1.forward(x)
        x = self.relu_1.forward(x)
        x = self.conv_layer_1.forward(x)

        x = self.batch_norm_2.forward(x)
        x = self.relu_2.forward(x)
        x = self.conv_layer_2.forward(x)

        x = self.batch_norm_3.forward(x)
        x = self.relu_3.forward(x)
        x = self.conv_layer_3.forward(x)

        x = self.squeeze_and_excitation.forward(x)

        x = x + identity

        return x


class SE_ResNetBottleneckBlock_layer_norm(nn.Module):

    def __init__(self, in_channels_block, act_map_resolution, is_downsampling_block = False):
        super(SE_ResNetBottleneckBlock_layer_norm, self).__init__()

        self.is_downsampling_block = is_downsampling_block
        self.in_channels_block = in_channels_block
        self.bottleneck_channels = in_channels_block // 4
        self.out_channels_block = self.bottleneck_channels * 4
        self.layer_1_stride = 1


        if self.is_downsampling_block:
            self.bottleneck_channels *= 2
            self.out_channels_block *= 2
            self.layer_1_stride = 2

            self.projection_shortcut = nn.Conv2d(
                in_channels = self.in_channels_block,
                out_channels = self.out_channels_block,
                kernel_size = 1,
                stride = 2,
                padding = 0
            )


        self.squeeze_and_excitation = SqueezeAndExcitationBlock(
            r = 16,
            channels = self.out_channels_block
        )

        layer_norm_shape = [self.in_channels_block] + act_map_resolution
        self.layer_norm_1 = nn.LayerNorm(layer_norm_shape)
        self.relu_1 = nn.ReLU()
        self.conv_layer_1 = nn.Conv2d(
            in_channels=self.in_channels_block,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            stride=self.layer_1_stride,
            padding=0)

        layer_norm_shape = [self.bottleneck_channels] + act_map_resolution
        self.layer_norm_2 = nn.LayerNorm(layer_norm_shape)
        self.relu_2 = nn.ReLU()
        self.conv_layer_2 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.bottleneck_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        layer_norm_shape = [self.bottleneck_channels] + act_map_resolution
        self.layer_norm_3 = nn.LayerNorm(layer_norm_shape)
        self.relu_3 = nn.ReLU()
        self.conv_layer_3 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.out_channels_block,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

    def forward(self,x):

        identity = x

        if self.is_downsampling_block:
            identity = self.projection_shortcut.forward(identity)

        x = self.layer_norm_1.forward(x)
        x = self.relu_1.forward(x)
        x = self.conv_layer_1.forward(x)

        x = self.layer_norm_2.forward(x)
        x = self.relu_2.forward(x)
        x = self.conv_layer_2.forward(x)

        x = self.layer_norm_3.forward(x)
        x = self.relu_3.forward(x)
        x = self.conv_layer_3.forward(x)

        x = self.squeeze_and_excitation.forward(x)

        x = x + identity

        return x


