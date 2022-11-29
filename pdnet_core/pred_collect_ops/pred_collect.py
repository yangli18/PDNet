import torch
import torch.nn as nn

from torch.autograd import Function

from . import pred_collect_ext


class CollectMergeFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                location,
                bias=None,
                channels_out=1,
                point_num=9,
                channels_offset=0,
                batch_proc=True):

        ctx.with_bias = bias is not None
        ctx.point_num = point_num
        ctx.channels_offset = channels_offset
        ctx.batch_proc = batch_proc
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if location.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, location, bias)
        output = input.new_empty((input.size(0), channels_out, location.size(2), location.size(3)))
        ctx._bufs = [input.new_empty(0)]
        pred_collect_ext.collect_merge_forward(
            input, bias, location, output,
            point_num, ctx.channels_offset, ctx.with_bias, ctx.batch_proc)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, location, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_location = torch.zeros_like(location)
        grad_bias = torch.zeros_like(bias)
        pred_collect_ext.collect_merge_backward(
            input, bias, ctx._bufs[0], location,
            grad_input, grad_bias,
            grad_location, grad_output,
            ctx.point_num, ctx.channels_offset, ctx.with_bias, ctx.batch_proc)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_location, grad_bias, None, None, None, None)


class CollectConcatFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                location,
                bias=None,
                channels_out=1,
                point_num=9,
                channels_offset=0,
                batch_proc=True):

        ctx.with_bias = bias is not None
        ctx.point_num = point_num
        ctx.channels_offset = channels_offset
        ctx.batch_proc = batch_proc
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if location.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, location, bias)
        output = input.new_empty((input.size(0), channels_out, location.size(2), location.size(3)))
        ctx._bufs = [input.new_empty(0)]
        pred_collect_ext.collect_concat_forward(
            input, bias, location, output,
            point_num, ctx.channels_offset, ctx.with_bias, ctx.batch_proc)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, location, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_location = torch.zeros_like(location)
        grad_bias = torch.zeros_like(bias)
        pred_collect_ext.collect_concat_backward(
            input, bias, ctx._bufs[0], location,
            grad_input, grad_bias,
            grad_location, grad_output,
            ctx.point_num, ctx.channels_offset, ctx.with_bias, ctx.batch_proc)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_location, grad_bias, None, None, None, None)


collect_merge = CollectMergeFunction.apply
collect_concat = CollectConcatFunction.apply


class CollectMerge(nn.Module):

    def __init__(self,
                 in_channels,
                 point_num,
                 batch_proc=True,
                 channels_offset=True,
                 bias=True):
        super(CollectMerge, self).__init__()
        self.in_channels = in_channels
        self.point_num = point_num
        self.batch_proc = batch_proc
        self.with_bias = bias
        assert in_channels % point_num == 0
        self.out_channels = in_channels // point_num
        self.channels_offset = self.out_channels if channels_offset else 0

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, location):
        return collect_merge(x, location, self.bias,
                             self.out_channels, self.point_num,
                             self.channels_offset, self.batch_proc)


class CollectConcat(nn.Module):

    def __init__(self,
                 in_channels,
                 point_num,
                 batch_proc=True,
                 channels_offset=True,
                 bias=True):
        super(CollectConcat, self).__init__()
        self.in_channels = in_channels
        self.point_num = point_num
        self.batch_proc = batch_proc
        self.with_bias = bias
        self.out_channels = in_channels if channels_offset else in_channels * point_num
        self.channels_offset = in_channels // point_num if channels_offset else 0

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, location):
        return collect_concat(x, location, self.bias,
                              self.out_channels, self.point_num,
                              self.channels_offset, self.batch_proc)