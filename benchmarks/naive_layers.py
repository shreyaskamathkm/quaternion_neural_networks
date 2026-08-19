import torch
import torch.nn as nn


class NaiveQuaternionGroupNorm2d(nn.Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on r,i,j,k part.
    '''

    def __init__(self, num_features,  channel_group_norm=0, eps=1e-5, affine=True):
        super().__init__()

        assert num_features % 4 == 0
        qfeatures = num_features//4

        # if num_groups > qfeatures//2:
        assert qfeatures % channel_group_norm == 0
        num_groups = qfeatures // channel_group_norm

        self.gn_r = torch.nn.GroupNorm(num_groups=num_groups, num_channels=qfeatures, eps=eps, affine=affine)
        self.gn_i = torch.nn.GroupNorm(num_groups=num_groups, num_channels=qfeatures, eps=eps, affine=affine)
        self.gn_j = torch.nn.GroupNorm(num_groups=num_groups, num_channels=qfeatures, eps=eps, affine=affine)
        self.gn_k = torch.nn.GroupNorm(num_groups=num_groups, num_channels=qfeatures, eps=eps, affine=affine)

    def forward(self, input):
        r, i, j, k = torch.chunk(input, 4, dim=1)
        r = self.gn_r(r)
        i = self.gn_i(i)
        j = self.gn_j(j)
        k = self.gn_k(k)
        return torch.cat((r, i, j, k), dim=1)


class NaiveQuaternionBatchNorm2d(nn.Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.bn(input)
