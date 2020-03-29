"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.nn import Linear, ModuleList
import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, is_relu_activ=False):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """
    super(MLP, self).__init__()

    self.is_relu_activ = is_relu_activ
    self.n_layers = len(n_hidden)

    self.list_hidden = ModuleList()

    if self.n_layers == 0:
      self.lastlay = Linear(n_inputs, n_classes)
    else:
      hidden_new = Linear(n_inputs, n_hidden[0])
      self.list_hidden.append(hidden_new)
      for i in range(self.n_layers-1):
        hidden_new = Linear(n_hidden[i], n_hidden[i+1])
        self.list_hidden.append(hidden_new)
      #end
      self.lastlay = Linear(n_hidden[-1], n_classes)


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
    for i in range(self.n_layers):
      if self.is_relu_activ:
        x = F.relu(self.list_hidden[i](x), inplace=True)
      else:
        x = self.list_hidden[i](x)
    #end

    return F.log_softmax(self.lastlay(x))
    #return out  # 'log_softmax' applied within 'CrossEntropyLoss()'
