from torch.nn import Module
import torch
from helper_functions import inspect_model
from torch.autograd import Variable

class BatchNormalization2D(Module):

    def __init__(self, num_features,  eps=1e-05, momentum=0.1):

        super(BatchNormalization2D, self).__init__()

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))

    def forward(self, x):

        batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True)
        batch_ch_std = torch.std(x, dim=(0,2,3), keepdim=True)

        if self.training:

            x = (x - batch_ch_mean) / batch_ch_std
            x = x * self.gamma + self.beta

        else:

            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

            self.running_avg_mean = self.running_avg_mean + self.alpha * (batch_ch_mean - self.running_avg_mean)
            self.running_avg_std = self.running_avg_std + self.alpha * (batch_ch_std - self.running_avg_std)

        return x


class BatchRenormalization2D(Module):

    def __init__(self, num_features,  eps=1e-05, momentum=0.1):
        super(BatchRenormalization2D, self).__init__()

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
        self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False)

        self.alpha = 0.01

        #TODO: Gradualy loosen up r and d during training
        self.r_max = 1.0
        self.d_max = 0.0

    def forward(self, x):

        batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True)
        batch_ch_std = torch.std(x, dim=(0,2,3), keepdim=True)

        if self.training:

            r = torch.clamp(batch_ch_mean / self.running_avg_mean, 1.0/self.r_max, self.r_max)
            d = torch.clamp(batch_ch_std / self.running_avg_std, -self.d_max, self.d_max)

            r.requires_grad = False
            d.requires_grad = False

            x = (x - batch_ch_mean) / batch_ch_std
            x = x * r + d
            x = self.gamma * x + self.beta

        else:

            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta


        self.running_avg_mean = self.running_avg_mean + self.alpha * (batch_ch_mean - self.running_avg_mean)
        self.running_avg_std = self.running_avg_std + self.alpha * (batch_ch_std - self.running_avg_std)

        return x


CHANNELS = 20
model_bre = BatchRenormalization2D(CHANNELS)
model_bn = BatchRenormalization2D(CHANNELS)

inspect_model(model_bn)

a = torch.rand((4,20,2,4))

x = model_bn.forward(a)

print(x)
