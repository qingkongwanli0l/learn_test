#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lzs
@file: learning_argparse.py
@time: 2022/12/20 0020 19:30
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvison.transform import ToTensor

# Download train data from open datasets
training_data = datasets.FashionMNIST(
     root='data',
     train=True,
     download=True,
     transform=ToTensor(),
 )