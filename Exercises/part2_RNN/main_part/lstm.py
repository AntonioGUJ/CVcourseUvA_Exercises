
"""
This module implements an LSTM in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, ModuleList

################################################################################


class LSTMcell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, device='cuda:0'):
      super(LSTMcell, self).__init__()

      self._input_size = input_size
      self._hidden_size = hidden_size
      self._batch_size = batch_size
      self._device = torch.device(device) if isinstance(device, str) else device

      # Initialize the weights and biases
      self._Wgx, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Wgh, self._bg = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wix, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Wih, self._bi = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wfx, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Wfh, self._bf = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wox, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Woh, self._bo = self.init_layer(self._hidden_size, self._hidden_size)

      # Weight initialization
      self.reset_parameters()

      # Cell state
      self.c = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)


    def forward(self, x, h):

      # modulation gate
      Wxx = torch.matmul(x, self._Wgx)
      Wxh = torch.matmul(h, self._Wgh) + self._bg
      g = torch.tanh(Wxx + Wxh)
      # input gate
      Wxx = torch.matmul(x, self._Wix)
      Wxh = torch.matmul(h, self._Wih) + self._bi
      i = torch.sigmoid(Wxx + Wxh)
      # forget gate
      Wxx = torch.matmul(x, self._Wfx)
      Wxh = torch.matmul(h, self._Wfh) + self._bf
      f = torch.sigmoid(Wxx + Wxh)
      # output gate
      Wxx = torch.matmul(x, self._Wox)
      Wxh = torch.matmul(h, self._Woh) + self._bo
      o = torch.sigmoid(Wxx + Wxh)
      # new cell state
      self.c = torch.mul(g, i) + torch.mul(self.c, f)
      # next hidden state
      out = torch.mul(torch.tanh(self.c), o)
      return out


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



# Recurrent neural network (many-to-one)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size, device='cuda:0', is_relu_activ=False):
      """
      Initializes LSTM object.

      Args:
        input_size: size of input vector.
        hidden_size: size of the hidden state.
        num_layers: how many layers for hidden or cell states.
        num_classes: number of output classes.

      TODO:
      Implement initialization of the network.
      IMPORTANT: This implementation does not work: requires to use 'retain_graph=True' when calling 'loss.backward()',
      otherwise the code crashes, but this causes the memory footprint to explode at some point during training
      """
      super(LSTM, self).__init__()

      self._input_size = input_size
      self._hidden_size = hidden_size
      self._num_layers = num_layers
      self._num_classes = num_classes
      self._batch_size = batch_size
      self._device = torch.device(device) if isinstance(device, str) else device
      self._is_relu_activ = is_relu_activ

      # Initialize the weights and biases
      self.lstm_lays = ModuleList()

      lstm_new = LSTMcell(self._input_size, self._hidden_size, self._batch_size, self._device)
      self.lstm_lays.append(lstm_new)
      for i in range(self._num_layers-1):
        lstm_new = LSTMcell(self._hidden_size, self._hidden_size, self._batch_size, self._device)
        self.lstm_lays.append(lstm_new)
      #end
      self._Wo, self._bo = self.init_layer(self._hidden_size, self._num_classes)

      # Weight initialization
      self.reset_parameters()


    def forward(self, x):
      """
      Performs forward pass of the input. Here an input tensor x.
      You are expected to do 3 steps:
       - set initial hidden and cell states
       - forward propagate LSTM
       - decode the hidden state of the last time step

      Args:
        x: input to the network
      Returns:
        out: outputs of the network

      TODO:
      Implement forward pass of the network.
      """

      x = x.permute(1, 0, 2)  # [seq_length, batch_size, 1]
      h_lays = []
      for i in range(self._num_layers):
          h_i = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)
          h_lays.append(h_i)
      # end

      # Step through time for LSTM
      for x_step in x:
        if self._is_relu_activ:
          h_lays[0] = F.relu(self.lstm_lays[0](x_step, h_lays[0]), inplace=True)
        else:
          h_lays[0] = self.lstm_lays[0](x_step, h_lays[0])

        # Loop through hidden LSTM cells
        for i in range(1,self._num_layers):
          if self._is_relu_activ:
            h_lays[i] = F.relu(self.lstm_lays[i](h_lays[i-1], h_lays[i]), inplace=True)
          else:
            h_lays[i] = self.lstm_lays[i](h_lays[i-1], h_lays[i])
        #end
      #end

      # Output layer
      out = torch.matmul(h_lays[-1], self._Wo) + self._bo
      #return F.log_softmax(out, dim=1)
      return out  #'log_softmax' applied within 'CrossEntropyLoss()'


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



# Recurrent neural network (many-to-one)
class LSTM_1layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size, device='cuda:0', is_relu_activ=False):
      """
      Initializes LSTM object. 

      Args:
        input_size: size of input vector.
        hidden_size: size of the hidden state.
        num_layers: how many layers for hidden or cell states.
        num_classes: number of output classes.

      TODO:
      Implement initialization of the network.
      """
      super(LSTM_1layer, self).__init__()

      self._input_size = input_size
      self._hidden_size = hidden_size
      self._num_layers = num_layers
      self._num_classes = num_classes
      self._batch_size = batch_size
      self._device = torch.device(device) if isinstance(device, str) else device
      self._is_relu_activ = is_relu_activ

      # Initialize the weights and biases
      self._Wgx, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Wgh, self._bg = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wix, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Wih, self._bi = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wfx, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Wfh, self._bf = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wox, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Woh, self._bo = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wph, self._bp = self.init_layer(self._hidden_size, self._num_classes)

      # Weight initialization
      self.reset_parameters()


    def forward(self, x):
      """
      Performs forward pass of the input. Here an input tensor x.
      You are expected to do 3 steps:
       - set initial hidden and cell states 
       - forward propagate LSTM
       - decode the hidden state of the last time step

      Args:
        x: input to the network
      Returns:
        out: outputs of the network

      TODO:
      Implement forward pass of the network.
      """

      x = x.permute(1, 0, 2)  # [seq_length, batch_size, 1]
      h = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)
      c = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)

      # Step through time for LSTM
      for x_step in x:
          # modulation gate
          Wxx = torch.matmul(x_step, self._Wgx)
          Wxh = torch.matmul(h, self._Wgh) + self._bg
          g = torch.tanh(Wxx + Wxh)
          # input gate
          Wxx = torch.matmul(x_step, self._Wix)
          Wxh = torch.matmul(h, self._Wih) + self._bi
          i = torch.sigmoid(Wxx + Wxh)
          # forget gate
          Wxx = torch.matmul(x_step, self._Wfx)
          Wxh = torch.matmul(h, self._Wfh) + self._bf
          f = torch.sigmoid(Wxx + Wxh)
          # output gate
          Wxx = torch.matmul(x_step, self._Wox)
          Wxh = torch.matmul(h, self._Woh) + self._bo
          o = torch.sigmoid(Wxx + Wxh)
          # cell state
          c = torch.mul(g, i) + torch.mul(c, f)
          # next hidden state
          if self._is_relu_activ:
              h = F.relu(torch.mul(torch.tanh(c), o), inplace=True)
          else:
              h = torch.mul(torch.tanh(c), o)
      #end

      # Output layer
      out = torch.matmul(h, self._Wph) + self._bp
      #return F.log_softmax(out, dim=1)
      return out  #'log_softmax' applied within 'CrossEntropyLoss()'


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