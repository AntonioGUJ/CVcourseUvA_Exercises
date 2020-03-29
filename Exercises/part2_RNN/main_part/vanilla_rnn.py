################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

################################################################################


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cuda:0', is_relu_activ=False):
        super(VanillaRNN, self).__init__()

        self._seq_length = seq_length
        self._input_dim = input_dim
        self._num_hidden = num_hidden
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._device = torch.device(device) if isinstance(device, str) else device

        # Initialize the weights and biases
        self._Wx, _ = self.init_layer(self._input_dim, self._num_hidden, use_bias=False)
        self._Wh, self._bh = self.init_layer(self._num_hidden, self._num_hidden)
        self._Wo, self._bo = self.init_layer(self._num_hidden, self._num_classes)

        # Weight initialization
        self.reset_parameters()


    def forward(self, x):

        x = x.permute(1,0,2)  # [seq_length, batch_size, 1]
        h = torch.zeros([self._batch_size, self._num_hidden])
        h = h.to(self._device)

        # Step through time for RNN
        for x_step in x:
            Wxx = torch.matmul(x_step, self._Wx)
            Wxh = torch.matmul(h, self._Wh) + self._bh
            h = torch.tanh(Wxx + Wxh)

        # Output layer
        #h = (h @ self._Wo) + self._bo  # not using python3.5, replace '@' by '.dot'
        h = torch.matmul(h, self._Wo) + self._bo
        #return F.log_softmax(h, dim=1)
        return h  #'log_softmax' applied within 'CrossEntropyLoss()'


    def init_layer(self, input_dim, output_dim, use_bias=True):
        # Basically implements nn.Linear
        W = Parameter(torch.empty(input_dim, output_dim, device=self._device, requires_grad=True))
        b = None
        if use_bias:
            b = Parameter(torch.empty(output_dim, device=self._device, requires_grad=True))
        return W, b

    def reset_parameters(self):
        stdv = 1.0/math.sqrt(self._num_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



class DeepRNN(nn.Module):
    # IMPORTANT: Does not work. Cannot declare operators within 'list_lays' for an arbitrary number of layers
    # The class cannot find the submodels when collecting them in '.parameters()'

    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size, device='cuda:0', is_relu_activ=False):
        super(DeepRNN, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._device = torch.device(device) if isinstance(device, str) else device
        self._is_relu_activ = is_relu_activ

        # Initialize the weights and biases
        self._Wx_lays = nn.ParameterList()
        self._Wh_lays = nn.ParameterList()
        self._bh_lays = nn.ParameterList()
        for i in range(self._num_layers):
            if i==0:
                self.Wx_i, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
            else:
                self.Wx_i, _ = self.init_layer(self._hidden_size, self._hidden_size, use_bias=False)
            self._Wx_lays.append(self.Wx_i)
            self.Wh_i, self.bh_i = self.init_layer(self._hidden_size, self._hidden_size)
            self._Wh_lays.append(self.Wh_i)
            self._bh_lays.append(self.bh_i)
            self._Wo, self._bo = self.init_layer(self._hidden_size, self._num_classes)
        #end

        # Weight initialization
        self.reset_parameters()


    def forward(self, x):

        x = x.permute(1,0,2)  # [seq_length, batch_size, 1]
        h_lays = []
        for i in range(self._num_layers):
            h_i = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)
            h_lays.append(h_i)
        #end

        # Step through time for RNN
        for x_step in x:
          # Loop through hidden LSTM cells
          for i in range(self._num_layers):
              if i==0:
                Wxx = torch.matmul(x_step, self._Wx_lays[i])
              else:
                Wxx = torch.matmul(h_lays[i-1], self._Wx_lays[i])
              Wxh = torch.matmul(h_lays[i], self._Wh_lays[i]) + self._bh_lays[i]
              # next hidden state
              if self._is_relu_activ:
                h_lays[i] = F.relu(torch.tanh(Wxx + Wxh), inplace=True)
              else:
                h_lays[i] = torch.tanh(Wxx + Wxh)
            #end
        #end

        # Output layer
        out = torch.matmul(h_lays[-1], self._Wo) + self._bo
        #return F.log_softmax(out, dim=1)
        return out  # 'log_softmax' applied within 'CrossEntropyLoss()'


    def init_layer(self, input_dim, output_dim, use_bias=True):
        # Basically implements nn.Linear
        W = Parameter(torch.empty(input_dim, output_dim, device=self._device, requires_grad=True))
        b = None
        if use_bias:
            b = Parameter(torch.empty(output_dim, device=self._device, requires_grad=True))
        return W, b

    def reset_parameters(self):
        stdv = 1.0/math.sqrt(self._hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

