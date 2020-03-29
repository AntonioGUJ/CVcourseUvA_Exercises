
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.optim.rmsprop import RMSprop
from torchsummary import summary

from main_part.dataset import PalindromeDataset
from main_part.vanilla_rnn import VanillaRNN, DeepRNN
from main_part.lstm import LSTM, LSTM_1layer

WORK_DIR_DEFAULT = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_3/'

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################


def accuracy_function(predictions, targets):
    predictions = F.softmax(predictions, dim=1)  #'F.log_softmax' already applied in model
    predictions = torch.argmax(predictions, dim=1)
    num_samples = predictions.shape[0]
    num_samples_equal = num_samples - torch.nonzero(predictions - targets).shape[0]
    return num_samples_equal / num_samples


def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device)
    elif config.model_type == 'LSTM':
        #model = LSTM(config.input_dim, config.num_hidden, config.num_layers, config.num_classes, config.batch_size, device)
        model = LSTM_1layer(config.input_dim, config.num_hidden, config.num_layers, config.num_classes, config.batch_size, device)
    # summary(model, (config.input_length, config.input_dim), batch_size=config.batch_size, device='cuda')

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), lr=config.learning_rate)


    # file to store loss history
    loss_history_file = os.path.join(config.work_dir, 'loss_history_{0}_T{1}.txt'.format(config.model_type, config.input_length))
    fout = open(loss_history_file, 'w')
    strheader = '/epoch/ /loss/ /accuracy/\n'
    fout.write(strheader)


    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # reshape batch_input to include 'input_dim' feats per elem
        batch_inputs = batch_inputs.reshape(batch_inputs.shape[0], config.input_length, config.input_dim)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        # Only for time measurement of step through network
        t1 = time.time()

        optimizer.zero_grad()
        # compute model output
        batch_predictions = model(batch_inputs)

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # compute loss
        lossfun = criterion(batch_predictions, batch_targets)
        # backprop algorithm step
        lossfun.backward()
        optimizer.step()
        loss = lossfun.item()
        accuracy = accuracy_function(batch_predictions, batch_targets)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss))

            # IMPORTANT: since data is randomly generated, there is differentiate between training / testing data
            # we can evaluate performance on prediction generated for the batch, if it's before doing backprop step
            # update loss history file
            strdata = '{0} {1:1.5} {2:1.5}\n'.format(step, loss, accuracy)
            fout.write(strdata)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    fout.close()


 ################################################################################
 ################################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in LSTM model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--work_dir', type=str, default=WORK_DIR_DEFAULT, help='Working directory to store output data')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Print all parsed entries in config
    for key, value in vars(config).items():
      print(key + ' : ' + str(value))

    if not os.path.exists(config.work_dir):
        os.makedirs(config.work_dir)

    # Train the model
    train(config)
