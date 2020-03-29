
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
from torch.nn import Parameter
from main_part.lstm import LSTMcell

################################################################################


# Recurrent neural network (many-to-one)
class LSTM_MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size, device='cuda:0', is_relu_activ=False):
      """
      Initializes LSTM_MNIST object. 

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
      super(LSTM_MNIST, self).__init__()

      self._input_size = input_size
      self._hidden_size = hidden_size
      self._num_layers = num_layers
      self._num_classes = num_classes
      self._batch_size = batch_size
      self._device = torch.device(device) if isinstance(device, str) else device
      self._is_relu_activ = is_relu_activ

      self.lstm1 = LSTMcell(self._input_size, self._hidden_size, self._batch_size, self._device)
      self.lstm2 = LSTMcell(self._hidden_size, self._hidden_size, self._batch_size, self._device)

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
      h1 = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)
      h2 = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)

      # Step through time for LSTM
      for x_step in x:
        if self._is_relu_activ:
          h1 = F.relu(self.lstm1(x_step, h1), inplace=True)
          h2 = F.relu(self.lstm2(h1, h2), inplace=True)
        else:
          h1 = self.lstm1(x_step, h1)
          h2 = self.lstm2(h1, h2)

      # Output layer
      out = torch.matmul(h2, self._Wo) + self._bo
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
        stdv = 1.0 / math.sqrt(self._hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



# Recurrent neural network (many-to-one)
class LSTM_MNIST_2layers(nn.Module):
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
      super(LSTM_MNIST_2layers, self).__init__()

      self._input_size = input_size
      self._hidden_size = hidden_size
      self._num_layers = num_layers
      self._num_classes = num_classes
      self._batch_size = batch_size
      self._device = torch.device(device) if isinstance(device, str) else device
      self._is_relu_activ = is_relu_activ

      # Initialize the weights and biases
      self._Wgx1, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Wgh1, self._bg1 = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wix1, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Wih1, self._bi1 = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wfx1, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Wfh1, self._bf1 = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wox1, _ = self.init_layer(self._input_size, self._hidden_size, use_bias=False)
      self._Woh1, self._bo1 = self.init_layer(self._hidden_size, self._hidden_size)

      self._Wgx2, _ = self.init_layer(self._hidden_size, self._hidden_size, use_bias=False)
      self._Wgh2, self._bg2 = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wix2, _ = self.init_layer(self._hidden_size, self._hidden_size, use_bias=False)
      self._Wih2, self._bi2 = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wfx2, _ = self.init_layer(self._hidden_size, self._hidden_size, use_bias=False)
      self._Wfh2, self._bf2 = self.init_layer(self._hidden_size, self._hidden_size)
      self._Wox2, _ = self.init_layer(self._hidden_size, self._hidden_size, use_bias=False)
      self._Woh2, self._bo2 = self.init_layer(self._hidden_size, self._hidden_size)

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
      h1 = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)
      h2 = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)
      c1 = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)
      c2 = torch.zeros([self._batch_size, self._hidden_size]).to(self._device)

      # Step through time for LSTM
      for x_step in x:
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
      #end

      # Output layer
      out = torch.matmul(h2, self._Wph) + self._bp
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