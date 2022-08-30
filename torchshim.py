###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
'''
Shims for PyTorch 1.8.1. Not needed in PyTorch 1.12.
'''
import torch  # pylint: disable=unused-import


def get_parameter(model: 'torch.nn.Module', target: str) -> 'torch.nn.Parameter':
    '''
    Returns the parameter given by ``target`` if it exists, otherwise throws an error.
    '''
    for name, param in model.named_parameters():
        if name == target:
            return param

    raise AttributeError(f'`{target}` is not an nn.Parameter')
