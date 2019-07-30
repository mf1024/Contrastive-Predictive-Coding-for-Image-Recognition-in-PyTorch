from torch.nn import Module
import torch

class BatchRenormalization2d(Module):

    def __init__(self, num_features,  eps=1e-05, momentum=0.1):

        alpha = torch.zeros(num_features)
        beta = torch.zeros(num_features)

    def forward(self, x):

        means = torch.mean(x, dim=(0,2,3))
        std = torch.std(x, dim=(0,2,3))

         



