import torch
import torch.nn as nn
from config.CSNet.CSNet_B import STEPS, ALPHA, VTH, TAU, Pt


class SpikeAct(torch.autograd.Function):  # CSNN
    alpha = ALPHA
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output_pos = torch.gt(input - VTH, 0)
        output_neg = torch.lt(input + VTH * Pt, 0)
        output = output_pos + output_neg * -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = abs(input) < SpikeAct.alpha
        hu = hu.float() / (2 * SpikeAct.alpha)
        return grad_input * hu


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = TAU * u_t_n1 * (1 - torch.abs(o_t_n1)) + W_mul_o_t1_n
    o_t1_n1 = SpikeAct.apply(u_t1_n1)
    return u_t1_n1, o_t1_n1


# class SpikeAct(torch.autograd.Function):  # SNN
#     alpha = ALPHA
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         output = torch.gt(input, 0)
#         return output.float()
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         hu = abs(input) < SpikeAct.alpha
#         hu = hu.float() / (2 * SpikeAct.alpha)
#         return grad_input * hu
#
#
# def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
#     u_t1_n1 = TAU * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
#     o_t1_n1 = SpikeAct.apply(u_t1_n1 - VTH)
#     return u_t1_n1, o_t1_n1


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None, steps=STEPS):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn
        self.steps = steps

    def forward(self, x):
        if isinstance(self.layer, nn.Conv2d):
            out_channels, padding, stride, kernel_size, dilate = \
            self.layer.out_channels, self.layer.padding, self.layer.stride, self.layer.kernel_size, self.layer.dilation
            in_channels, in_H, in_W = x.shape[-5], x.shape[-3], x.shape[-2]
            out_H = (in_H + 2 * padding[0] - dilate[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
            out_W = (in_W + 2 * padding[1] - dilate[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
            x_ = torch.zeros([in_channels, out_channels, int(out_H), int(out_W), self.steps], device=x.device)

        elif isinstance(self.layer, nn.MaxPool2d):
            padding, stride, kernel_size, dilate = self.layer.padding, self.layer.stride, self.layer.kernel_size, self.layer.dilation
            in_channels, out_channels, in_H, in_W = x.shape[-5], x.shape[-4], x.shape[-3], x.shape[-2]
            out_H = (in_H + 2 * padding - dilate * (kernel_size - 1) - 1) / stride + 1
            out_W = (in_W + 2 * padding - dilate * (kernel_size - 1) - 1) / stride + 1
            x_ = torch.zeros([in_channels, out_channels, int(out_H), int(out_W), self.steps], device=x.device)

        for step in range(self.steps):
            x_[..., step] = self.layer(x[..., step].float())
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class LIFSpike(nn.Module):
    def __init__(self, steps=STEPS):
        super(LIFSpike, self).__init__()
        self.steps = steps

    def forward(self, x):
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(self.steps):
            u, out[..., step] = state_update(u, out[..., max(step - 1, 0)], x[..., step])
        return out


class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = self.alpha * VTH * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input

