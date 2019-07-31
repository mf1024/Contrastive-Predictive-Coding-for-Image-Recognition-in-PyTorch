from torch.nn import Module
import torch
from torch.autograd import Variable

class BatchNormalization2D(Module):

    def __init__(self, num_features,  eps=1e-05, momentum=0.1):

        super(BatchNormalization2D, self).__init__()

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))

        self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
        self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False)

        self.alpha = torch.tensor( (0.01), requires_grad = False)

    def forward(self, x):

        device = self.gamma.device

        batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
        batch_ch_std = torch.std(x, dim=(0,2,3), keepdim=True).to(device)

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)
        self.alpha = self.alpha.to(device)

        if self.training:

            x = (x - batch_ch_mean) / batch_ch_std
            x = x * self.gamma + self.beta

        else:

            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        self.running_avg_mean = self.running_avg_mean + self.alpha * (batch_ch_mean.data.to(device) - self.running_avg_mean)
        self.running_avg_std = self.running_avg_std + self.alpha * (batch_ch_std.data.to(device) - self.running_avg_std)

        return x


class BatchRenormalization2D(Module):

    def __init__(self, num_features,  eps=1e-05, momentum=0.1):
        super(BatchRenormalization2D, self).__init__()

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
        self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False)

        self.alpha = torch.tensor( (0.01), requires_grad = False)

        #TODO: Gradualy loosen up r and d during training
        # self.r_max = torch.tensor( (1.1), requires_grad = False)
        # self.d_max = torch.tensor( (0.1), requires_grad = False)

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = 0.001
        self.d_max_inc_step = 0.001

        self.r_max = torch.tensor( (1.0), requires_grad = False)
        self.d_max = torch.tensor( (0.0), requires_grad = False)

    def forward(self, x):

        device = self.gamma.device

        batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
        batch_ch_std = torch.std(x, dim=(0,2,3), keepdim=True).to(device)

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)
        self.alpha = self.alpha.to(device)

        self.r_max = self.r_max.to(device)
        self.d_max = self.d_max.to(device)


        if self.training:

            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).to(device).data.to(device)
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max, self.d_max).to(device).data.to(device)

            x = (x - batch_ch_mean) / batch_ch_std
            x = x * r + d
            x = self.gamma * x + self.beta

            if self.r_max < self.max_r_max:
                self.r_max += self.r_max_inc_step

            if self.d_max < self.max_d_max:
                self.d_max += self.d_max_inc_step

        else:

            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        self.running_avg_mean = self.running_avg_mean + self.alpha * (batch_ch_mean.data.to(device) - self.running_avg_mean)
        self.running_avg_std = self.running_avg_std + self.alpha * (batch_ch_std.data.to(device) - self.running_avg_std)

        return x

