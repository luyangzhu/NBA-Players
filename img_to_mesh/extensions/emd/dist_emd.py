import math
from torch import nn
from torch.autograd import Function
import torch
import sys
from numbers import Number
from collections import Set, Mapping, deque
import emd


# Earth Mover distance module
# GPU tensors only
class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        match = torch.zeros(batchsize, n , m).cuda()
        cost = torch.zeros(batchsize, ).cuda()
        temp = torch.zeros(batchsize, 2 * (m+n)).cuda()
        emd.forward(xyz1, xyz2, match, cost, temp)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, gradcost):
        xyz1, xyz2, match = ctx.saved_tensors
        gradxyz1 = torch.zeros(xyz1.size()).cuda()
        gradxyz2 = torch.zeros(xyz2.size()).cuda()
        emd.backward(xyz1, xyz2, gradxyz1, gradxyz2, match)
        return gradxyz1, gradxyz2

class emdDist(nn.Module):
    def __init__(self):
        super(emdDist, self).__init__()

    def forward(self, input1, input2):
        return emdFunction.apply(input1, input2)

