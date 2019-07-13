from resnet_blocks import ResNetBottleneckBlock
import torch
from torch import nn
from torch.nn import Module

class ResEncoderModel(Module):
    def __init__(self):
        super(ResEncoderModel, self).__init__()

        # TODO: ResNet v2
        # TODO: Squeeze and Excitation

        # Input is 3 x 64 x 64
        # prep -> 256 x 32 x 32

        self.conv_blocks = [10,10,10] #256x32x32 -> 512x16x16 -> 1024x8x8
        self.num_blocks = len(self.conv_blocks)
        self.start_channels = 256


        self.prep = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = self.start_channels,
                kernel_size = 7,
                stride = 1, #Let's not reduce twice
                padding = 3
            ),
            nn.BatchNorm2d(
                num_features = self.start_channels,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        )
        # Output 256x 32 x 32
        current_channels = self.start_channels

        self.resnet_blocks = nn.ModuleList()

        for block_idx, conv_block_num in enumerate(self.conv_blocks):
            resnet_block = nn.Sequential()

            for conv_block_idx in range(conv_block_num):

                is_downsampling_block = False

                if block_idx > 0 and conv_block_idx == 0:
                    is_downsampling_block = True

                resnet_block.add_module(
                    f'conv_{conv_block_idx}',
                    ResNetBottleneckBlock(
                        in_channels_block = current_channels,
                        is_downsampling_block = is_downsampling_block
                    )
                )

                if is_downsampling_block:
                    current_channels *= 2

            self.resnet_blocks.append(resnet_block)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)


    def forward(self, x):

        x = self.prep.forward(x)
        #print(f'shape after prep {x.shape}')
        for i in range(self.num_blocks):
            x = self.resnet_blocks[i].forward(x)
            #print(f'shape after resnet_block {i} {x.shape}')

        #print(f'shape after resnet {x.shape}')
        x = self.avg_pool.forward(x)
        x = torch.squeeze(x, dim=3)
        x = torch.squeeze(x, dim=2)
        ##print(f'shape after avg_pool {x.shape}')

        return x


class ContextPredictionModel(Module):

    def __init__(self, in_channels):
        super(ContextPredictionModel, self).__init__()

        self.in_channels = in_channels

        # Input will be 1024x7x7

        # Two sets of convolutional context networks - one for vertical, one for horizontal agregation.

        # Prediction 3 steps ahead. So I will have 8 outputs.
        # [0,2:6] predict->[3,4,5:6],[1,3:6] predict->[4,5,6:6]
        # [4,6:6] predict->[3,2,1:6],[3,5:6] predict->[2,1,0:6]

        # [6:0,2] predict->[6:3,4,5],[6:1,3] predict->[6:4,5,6]
        # [6:4,6] predict->[6:3,2,1],[6:3,5] predict->[6:2,1,0]

        self.context_layers = 3
        self.context_conv = nn.Sequential()

        for layer_idx in range(self.context_layers):
            self.context_conv.add_module(f'batch_norm_{layer_idx}',nn.BatchNorm2d(self.in_channels)),
            self.context_conv.add_module(f'relu_{layer_idx}',nn.ReLU())
            self.context_conv.add_module(
                f'conv2d_{layer_idx}',
                nn.Conv2d(
                    in_channels = self.in_channels,
                    out_channels = self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

        self.context_conv.add_module(
            'adaptive_avg_pool',
            nn.AdaptiveAvgPool2d(output_size=1)
        )


        # UP - 0, RIGHT = 1, DOWN = 2, LEFT = 3

        self.prediction_weights = nn.ModuleList([nn.ModuleList() for i in range(4)])

        for direction in range(4):
            for prediction_steps in range(3):
                self.prediction_weights[direction].append(
                    nn.Linear(
                        in_features = self.in_channels,
                        out_features = self.in_channels,
                    )
                )


    def forward(self, x):

        prediction_list = []

        for y1 in range(5):
            for x1 in range(5):
                y2 = y1 + 2
                x2 = x1 + 2

                context_input = x[:,:,y1:y2+1,x1:x2+1]
                #print(f"x1:{x1} y1:{y1}, x2:{x2}, y2:{y2}")

                #print(f"context_input shape {context_input.shape}")
                context = self.context_conv.forward(context_input)

                #print(f"context shape {context.shape}")

                context = context.squeeze(dim=3)
                context = context.squeeze(dim=2)
                #print(f"context shape {context.shape}")

                # UP - 0, RIGHT = 1, DOWN = 2, LEFT = 3

                #predict down

                if y2 == 2 or y2 == 3:
                    for steps_y_plus in range(3):
                        y3 = y2 + (steps_y_plus+1)
                        if y3 > 6:
                            break

                        # print(f"y+ pred for y1:{y1} y2:{y2} x1:{x1} x2:{x2} y3:{y3}")

                        prediction = self.prediction_weights[0][steps_y_plus].forward(context)
                        #print(f"predfiction shape on context {prediction.shape}")
                        prediction_list.append(
                            dict(
                                y=y3,
                                x=x1+1,
                                prediction=prediction,
                                pred_class=f"y+{steps_y_plus+1}",
                            )
                        )


                if y1 == 3 or y1 == 4:
                    for steps_y_minus in range(3):
                        y3 = y1 - (steps_y_minus+1)
                        if y3 < 0:
                            break

                        # print(f"y- pred for y1:{y1} y2:{y2} x1:{x1} x2:{x2} y3:{y3}")

                        prediction = self.prediction_weights[1][steps_y_minus].forward(context)
                        #print(f"predfiction shape on context {prediction.shape}")
                        prediction_list.append(
                            dict(
                                y=y3,
                                x=x1+1,
                                prediction=prediction,
                                pred_class=f"y-{steps_y_minus+1}",
                            )
                        )


                if x2 == 2 or x2 == 3:
                    for steps_x_plus in range(3):
                        x3 = x2 + (steps_x_plus+1)
                        if x3 > 6:
                            break

                        # print(f"x+ pred for y1:{y1} y2:{y2} x1:{x1} x2:{x2} x3:{x3}")
                        prediction = self.prediction_weights[2][steps_x_plus].forward(context)
                        #print(f"predfiction shape on context {prediction.shape}")
                        prediction_list.append(
                            dict(
                                y=y1+1,
                                x=x3,
                                prediction=prediction,
                                pred_class=f"x+{steps_x_plus+1}",
                            )
                        )


                if x1 == 3 or x1 == 4:
                    for steps_x_minus in range(3):
                        x3 = x1 - (steps_x_minus+1)
                        if x3 < 0:
                            break

                        # print(f"x- pred for y1:{y1} y2:{y2} x1:{x1} x2{x2} x3:{x3}")
                        prediction = self.prediction_weights[3][steps_x_minus].forward(context)
                        #print(f"predfiction shape on context {prediction.shape}")
                        prediction_list.append(
                            dict(
                                y=y1+1,
                                x=x3,
                                prediction=prediction,
                                pred_class=f"x-{steps_x_minus+1}",
                            )
                        )

        return prediction_list


class ResClassificatorModel(Module):

    def __init__(self, in_channels, num_classes):
        super(ResClassificatorModel, self).__init__()

        # Input is [Bx1024x7x7] shaped
        # Input is [Bxinput_channelsx7x7] shaped tensor

        self.num_classes = num_classes
        self.num_res_blocks = 11
        self.in_channels = in_channels
        self.channels = 1024

        self.prep = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = self.channels,
                kernel_size = 1,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(num_features=self.channels),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential()
        for i in range(self.num_res_blocks-1):
            self.res_blocks.add_module(
                f'res_block_{i}',
                ResNetBottleneckBlock(
                    in_channels_block = self.channels
                )
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(
            in_features = self.channels,
            out_features = self.num_classes
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.prep(x)
        x = self.res_blocks(x)
        x = self.avg_pool(x)
        x = x.squeeze(dim=3)
        x = x.squeeze(dim=2)
        x = self.linear(x)
        x = self.softmax(x)

        return x
