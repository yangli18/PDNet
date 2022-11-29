# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from pdnet_core import _C

from apex import amp

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)
ml_nms = _C.ml_nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
