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

import os
import time
from datetime import datetime
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from bonus_part2.dataset import TextDataset
from bonus_part2.model import TextGenerationModel

WORK_DIR_DEFAULT = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_3_bonus2/'

################################################################################


def compute_string_predictions(fun_convert_to_string, inputs, targets, predictions):
    predictions = F.softmax(predictions, dim=1)  #'F.log_softmax' already applied in model
    predictions = torch.argmax(predictions, dim=1)
    inputs = inputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    string_inputs = []
    string_targets = []
    string_predictions = []
    batch_size = predictions.shape[0]
    for i in range(batch_size):
        new_string_input = fun_convert_to_string(inputs[i].squeeze())
        string_inputs.append(new_string_input)
        new_string_targets = fun_convert_to_string([targets[i]])
        string_targets.append(new_string_targets)
        new_string_predictions = fun_convert_to_string([predictions[i]])
        string_predictions.append(new_string_predictions)
    #end
    return string_inputs, string_targets, string_predictions


def loss_function(criterion, predictions, targets):
    sequence_length = predictions.shape[0]
    losses = []
    for i in range(sequence_length):
        losses.append(criterion(predictions[i], targets[i]))
    #end

def accuracy_function(predictions, targets):
    predictions = F.softmax(predictions, dim=1)  #'F.log_softmax' already applied in model
    predictions = torch.argmax(predictions, dim=1)
    num_samples = predictions.shape[0]
    num_samples_equal = num_samples - torch.nonzero(predictions - targets).shape[0]
    return num_samples_equal / num_samples


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    vocab_size = dataset.vocab_size

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers, device=device, is_relu_activ=False)
    # summary(model, (config.seq_length, 1), batch_size=config.batch_size, device='cuda')
    model.train()

    # Setup the loss and optimizer
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)


    # file to store loss history
    loss_history_file = os.path.join(config.work_dir, 'loss_history.txt')
    fout = open(loss_history_file, 'w')
    strheader = '/epoch/ /loss/ /accuracy/\n'
    fout.write(strheader)


    globstep = 0
    total_step = len(data_loader)
    max_loops_book = 100

    for k in range(max_loops_book):
        #reuse text from book if the end is reached
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            if step==total_step-1:
                break

            # reshape batch_input to include 'input_dim=1' feats per elem
            batch_inputs = batch_inputs.reshape(batch_inputs.shape[0], config.seq_length, 1)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_targets_last = batch_targets[:, -1]

            # Only for time measurement of step through network
            t1 = time.time()

            optimizer.zero_grad()
            # compute model output
            batch_predictions = model(batch_inputs)

            # Clip gradients of parameters to prevent the issue of exploding gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

            # compute loss
            lossfun = criterion(batch_predictions, batch_targets_last)
            # backprop algorithm step
            lossfun.backward()
            optimizer.step()
            loss = lossfun.item()
            accuracy = accuracy_function(batch_predictions, batch_targets_last)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1)

            if globstep % config.print_every == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), globstep,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss))

                # update loss history file
                strdata = '{0} {1:1.5} {2:1.5}\n'.format(globstep, loss, accuracy)
                fout.write(strdata)

            if globstep % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                batch_string_inputs, \
                batch_string_targets_last, \
                batch_string_predictions = compute_string_predictions(dataset.convert_to_string, batch_inputs,
                                                                      batch_targets_last, batch_predictions)

                print("Show predictions of {} samples phrases:".format(config.batch_size))
                for i in range(config.batch_size):
                    # remove predicted 'end line \n' to better visualize
                    string_inputs = batch_string_inputs[i].replace('\n', ' ').replace('\r', ' ')
                    string_targets_last = batch_string_targets_last[i].replace('\n', ' ').replace('\r', ' ')
                    string_predictions = batch_string_predictions[i].replace('\n', ' ').replace('\r', ' ')

                    print("{0}: {1} '{2}'[{3}]".format(i, string_inputs, string_predictions, string_targets_last))
                # end

            if globstep == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

            globstep += 1
        #end
    #end

    print('Done training.')
    fout.close()


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

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
