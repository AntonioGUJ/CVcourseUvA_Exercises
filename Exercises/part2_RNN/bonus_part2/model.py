# MIT License
#
# Copyright (c) 2017 Tom Runia
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


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, lstm_num_hidden=256,
                 lstm_num_layers=2, device='cuda:0', is_relu_activ=False):

        super(TextGenerationModel, self).__init__()
        # Initialization here...

        self._batch_size = batch_size
        self._seq_length = seq_length
        self._input_size = 1
        self._vocabulary_size = vocabulary_size
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._device = torch.device(device) if isinstance(device, str) else device
        self._is_relu_activ = is_relu_activ

        # Initialize the weights and biases
        self._Wgx1, _ = self.init_layer(self._input_size, self._lstm_num_hidden, use_bias=False)
        self._Wgh1, self._bg1 = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden)
        self._Wix1, _ = self.init_layer(self._input_size, self._lstm_num_hidden, use_bias=False)
        self._Wih1, self._bi1 = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden)
        self._Wfx1, _ = self.init_layer(self._input_size, self._lstm_num_hidden, use_bias=False)
        self._Wfh1, self._bf1 = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden)
        self._Wox1, _ = self.init_layer(self._input_size, self._lstm_num_hidden, use_bias=False)
        self._Woh1, self._bo1 = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden)

        self._Wgx2, _ = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden, use_bias=False)
        self._Wgh2, self._bg2 = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden)
        self._Wix2, _ = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden, use_bias=False)
        self._Wih2, self._bi2 = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden)
        self._Wfx2, _ = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden, use_bias=False)
        self._Wfh2, self._bf2 = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden)
        self._Wox2, _ = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden, use_bias=False)
        self._Woh2, self._bo2 = self.init_layer(self._lstm_num_hidden, self._lstm_num_hidden)

        self._Wph, self._bp = self.init_layer(self._lstm_num_hidden, self._vocabulary_size)

        # Weight initialization
        self.reset_parameters()


    def forward(self, x):
        # Implementation here...

        x = x.permute(1, 0, 2)  # [seq_length, batch_size, 1]
        h1 = torch.zeros([self._batch_size, self._lstm_num_hidden]).to(self._device)
        h2 = torch.zeros([self._batch_size, self._lstm_num_hidden]).to(self._device)
        c1 = torch.zeros([self._batch_size, self._lstm_num_hidden]).to(self._device)
        c2 = torch.zeros([self._batch_size, self._lstm_num_hidden]).to(self._device)

        #out = torch.zeros([self._seq_length, self._batch_size, self._vocabulary_size]).to(self._device)

        # Step through time for LSTM
        for i, x_step in enumerate(x):
            # **************
            # LSTM layer 1 *
            # **************
            # modulation gate
            Wxx = torch.matmul(x_step, self._Wgx1)
            Wxh = torch.matmul(h1, self._Wgh1) + self._bg1
            g = torch.tanh(Wxx + Wxh)
            # input gate
            Wxx = torch.matmul(x_step, self._Wix1)
            Wxh = torch.matmul(h1, self._Wih1) + self._bi1
            i = torch.sigmoid(Wxx + Wxh)
            # forget gate
            Wxx = torch.matmul(x_step, self._Wfx1)
            Wxh = torch.matmul(h1, self._Wfh1) + self._bf1
            f = torch.sigmoid(Wxx + Wxh)
            # output gate
            Wxx = torch.matmul(x_step, self._Wox1)
            Wxh = torch.matmul(h1, self._Woh1) + self._bo1
            o = torch.sigmoid(Wxx + Wxh)
            # cell state
            c1 = torch.mul(g, i) + torch.mul(c1, f)
            # next hidden state
            if self._is_relu_activ:
                h1 = F.relu(torch.mul(torch.tanh(c1), o), inplace=True)
            else:
                h1 = torch.mul(torch.tanh(c1), o)

            # **************
            # LSTM layer 2 *
            # **************
            # modulation gate
            Wxx = torch.matmul(h1, self._Wgx2)
            Wxh = torch.matmul(h2, self._Wgh2) + self._bg2
            g = torch.tanh(Wxx + Wxh)
            # input gate
            Wxx = torch.matmul(h1, self._Wix2)
            Wxh = torch.matmul(h2, self._Wih2) + self._bi2
            i = torch.sigmoid(Wxx + Wxh)
            # forget gate
            Wxx = torch.matmul(h1, self._Wfx2)
            Wxh = torch.matmul(h2, self._Wfh2) + self._bf2
            f = torch.sigmoid(Wxx + Wxh)
            # output gate
            Wxx = torch.matmul(h1, self._Wox2)
            Wxh = torch.matmul(h2, self._Woh2) + self._bo2
            o = torch.sigmoid(Wxx + Wxh)
            # cell state
            c2 = torch.mul(g, i) + torch.mul(c2, f)
            # next hidden state
            if self._is_relu_activ:
                h2 = F.relu(torch.mul(torch.tanh(c2), o), inplace=True)
            else:
                h2 = torch.mul(torch.tanh(c2), o)

            ## Output layer
            #out[i] = torch.matmul(h2, self._Wph) + self._bp
            ## 'log_softmax' applied within 'CrossEntropyLoss()'
            ##out[i] = F.log_softmax(torch.matmul(h2, self._Wph) + self._bp, dim=1)
        # end

        # Output layer
        out = torch.matmul(h2, self._Wph) + self._bp
        # return F.log_softmax(out, dim=1)
        return out  # 'log_softmax' applied within 'CrossEntropyLoss()'


    def init_layer(self, input_dim, output_dim, use_bias=True):
        # Basically implements nn.Linear
        W = Parameter(torch.empty(input_dim, output_dim, device=self._device, requires_grad=True))
        b = None
        if use_bias:
            b = Parameter(torch.empty(output_dim, device=self._device, requires_grad=True))
        return W, b

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self._lstm_num_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)