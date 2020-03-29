"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.nn import Conv2d, MaxPool2d, AvgPool2d, Linear, BatchNorm2d, Dropout2d
import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvBlock(nn.Module):

  def __init__(self, n_feat_in, n_feat_out, is_dropout=False, dropout_rate=0.5, is_batchnorm=False, is_relu_activ=True):
    super(ConvBlock, self).__init__()

    self.is_dropout = is_dropout
    self.is_batchnorm = is_batchnorm
    self.is_relu_activ = is_relu_activ

    self.convlay = Conv2d(n_feat_in, n_feat_out, kernel_size= 3, padding= 1)
    if is_dropout:
      self.dropout = Dropout2d(dropout_rate)
    if is_batchnorm:
      self.batchnorm = BatchNorm2d(n_feat_out)

  def forward(self, x):

    x = self.convlay(x)
    if self.is_dropout:
      x = self.dropout(x)
    if self.is_batchnorm:
      x = self.batchnorm(x)
    if self.is_relu_activ:
      return F.relu(x, inplace=True)
    else:
      return x


class ConvBlockVGGnet(nn.Module):

  def __init__(self, n_feat_in, n_feat_out):
    super(ConvBlockVGGnet, self).__init__()

    self.convlay = Conv2d(n_feat_in, n_feat_out, kernel_size= 3, padding= 1)
    self.batchnorm = BatchNorm2d(n_feat_out)

  def forward(self, x):

    return F.relu( self.batchnorm( self.convlay(x)), inplace=True)


class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, size_image, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    
    TODO:
    Implement initialization of the network.
    """
    super(ConvNet, self).__init__()

    n_feat_lay1 = 64
    self.conv1 = ConvBlockVGGnet(n_channels, n_feat_lay1)
    self.maxpool1 = MaxPool2d(kernel_size= 2)

    n_feat_lay2 = 2 * n_feat_lay1
    self.conv2 = ConvBlockVGGnet(n_feat_lay1, n_feat_lay2)
    self.maxpool2 = MaxPool2d(kernel_size= 2, padding= 0)

    n_feat_lay3 = 2 * n_feat_lay2
    self.conv3_a = ConvBlockVGGnet(n_feat_lay2, n_feat_lay3)
    self.conv3_b = ConvBlockVGGnet(n_feat_lay3, n_feat_lay3)
    self.maxpool3 = MaxPool2d(kernel_size= 2, padding= 0)

    n_feat_lay4 = 2 * n_feat_lay3
    self.conv4_a = ConvBlockVGGnet(n_feat_lay3, n_feat_lay4)
    self.conv4_b = ConvBlockVGGnet(n_feat_lay4, n_feat_lay4)
    self.maxpool4 = MaxPool2d(kernel_size= 2, padding= 0)

    n_feat_lay5 = n_feat_lay4
    self.conv5_a = ConvBlockVGGnet(n_feat_lay4, n_feat_lay5)
    self.conv5_b = ConvBlockVGGnet(n_feat_lay5, n_feat_lay5)
    self.maxpool5 = MaxPool2d(kernel_size= 2, padding= 0)

    self.avgpool = AvgPool2d(kernel_size= 1, padding= 0)

    size_img_out = ( int(size_image[0]/32), int(size_image[1]/32) ) # image size after 4 poolings
    n_feat_inlin = size_img_out[0] * size_img_out[1] * n_feat_lay5
    self.lastlay = Linear(n_feat_inlin, n_classes)


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.maxpool2(x)
    x = self.conv3_a(x)
    x = self.conv3_b(x)
    x = self.maxpool3(x)
    x = self.conv4_a(x)
    x = self.conv4_b(x)
    x = self.maxpool4(x)
    x = self.conv5_a(x)
    x = self.conv5_b(x)
    x = self.maxpool5(x)
    x = self.avgpool(x)

    out = self.lastlay(x.flatten(start_dim=1))
    return F.log_softmax(out, dim=1)
    #return out  # 'log_softmax' applied within 'CrossEntropyLoss()'
