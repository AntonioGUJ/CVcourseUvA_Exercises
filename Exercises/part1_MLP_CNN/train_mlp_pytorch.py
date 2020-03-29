"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
from cifar10_utils import get_cifar10
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from datetime import datetime as dt
from tqdm import tqdm


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
IS_RELU_ACTIV_DEFAULT = True
LEARNING_RATE_DEFAULT = 0.001
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
IS_IMPL_CUDA = True

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/code/part1_MLP_CNN/cifar10/cifar-10-batches-py'
WORK_DIR_DEFAULT = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_1/'
FLAGS = None


def accuracy_function(predictions, targets, is_one_hot=True):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  #predictions = F.softmax(predictions, dim=1)  #'F.log_softmax' already applied in model
  predictions = torch.argmax(predictions, dim=1)
  if is_one_hot:
    targets = torch.argmax(targets, dim=1)
  num_samples = predictions.shape[0]
  num_samples_equal = num_samples - torch.nonzero(predictions - targets).shape[0]
  return num_samples_equal / num_samples


def convert_data_torch(X_batch, Y_batch, device):
  if device == 'cuda:0':
    return (torch.from_numpy(X_batch).type(torch.cuda.FloatTensor).to(device),
            torch.from_numpy(Y_batch).type(torch.cuda.LongTensor).to(device))
  else:
    return (torch.from_numpy(X_batch).type(torch.FloatTensor),
            torch.from_numpy(Y_batch).type(torch.LongTensor))


def train():
  """
  Performs training and evaluation of MLP model.

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  learning_rate = FLAGS.learning_rate
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  eval_freq = FLAGS.eval_freq
  is_relu_activ = FLAGS.is_relu_activ
  data_dir = FLAGS.data_dir
  work_dir = FLAGS.work_dir
  _device = FLAGS.device

  # ***********
  # LOAD DATA *
  # ***********
  # use test data as validation data
  all_data = get_cifar10(data_dir, one_hot=False, validation_size=0)
  train_data = all_data['train']
  test_data = all_data['test']

  train_data.convert_data_to_MLP_input(is_one_hot=False)
  test_data.convert_data_to_MLP_input(is_one_hot=False)

  # *************
  # BUILD MODEL *
  # *************
  n_feat_in = train_data.images[0].size
  n_classes = 10 #train_data.labels[0].size

  # design model (and compiled in cuda)
  model = MLP(n_inputs=n_feat_in, n_hidden=dnn_hidden_units, n_classes=n_classes, is_relu_activ=is_relu_activ)
  model.to(_device)
  summary(model, (n_feat_in,), batch_size=batch_size, device='cuda' if _device=='cuda:0' else 'cpu')

  # design optimizer
  #optimizer = Adam(model.parameters(), lr=learning_rate)
  optimizer = SGD(model.parameters(), lr=learning_rate)
  # design loss function
  loss_fun = nn.NLLLoss()
  #loss_fun = nn.CrossEntropyLoss()


  # file to store loss history
  loss_history_file = os.path.join(work_dir, 'loss_history_dnn-{}_{}.txt'.format('-'.join([str(s) for s in dnn_hidden_units]),
                                                                                  'relu' if is_relu_activ else 'norelu'))
  fout = open(loss_history_file, 'w')
  strheader = '/epoch/ /loss/ /valid_loss/ /accuracy/ /valid_accuracy/\n'
  fout.write(strheader)


  print('START TRAINING...\n')

  for i_epoch in range(0, max_steps):
    # ***********************************
    # RUN THROUGH TRAIN DATA IN I_EPOCH *
    # ***********************************
    model.train() # mode 'train' model

    #progressbar = tqdm(total= train_data.num_batches(batch_size),
    #                   desc= 'Epochs {}/{}'.format(i_epoch, max_steps),
    #                   bar_format= '{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]')
    time_ini = dt.now()
    sumrun_loss = 0.0
    sumrun_accuracy = 0.0

    # loop through train data in i_epoch
    check_epoch_completed = train_data.epochs_completed
    epoch_completed = train_data.epochs_completed
    i_batch = 0
    while epoch_completed == check_epoch_completed:
      # retrieve data batch
      X_batch, Y_batch = train_data.next_batch(batch_size)
      # convert data to torch format, and transfer to GPU
      X_batch, Y_batch = convert_data_torch(X_batch, Y_batch, device=_device)

      optimizer.zero_grad()
      # compute model output
      pred_batch = model(X_batch)
      # compute loss
      loss = loss_fun(pred_batch, Y_batch)
      # backprop algorithm step
      loss.backward()
      optimizer.step()

      sumrun_loss += loss.item()
      # compute accuracy
      accuracy = accuracy_function(pred_batch, Y_batch, is_one_hot=False)
      sumrun_accuracy += accuracy

      #loss_partial = sumrun_loss/(i_batch+1)
      #progressbar.set_postfix(loss='{0:1.5f}'.format(loss_partial))
      #progressbar.update(1)

      i_batch += 1
      #epoch_completed = train_data.epochs_completed
      epoch_completed = -1 # train only batch data per epoch
    #end

    train_loss = sumrun_loss/i_batch
    train_accuracy = sumrun_accuracy/i_batch

    if i_epoch % 10 == 0:
      time_now = dt.now()
      print('Train epoch {0}/{1}, Batch Size = {2}. Train Loss = {3:1.5}. Accuracy = {4:1.5}. Time compute = {5}'.
            format(i_epoch, max_steps, batch_size, train_loss, train_accuracy, (time_now - time_ini).seconds))


    # *************************************
    # RUN THROUGH TESTING DATA IN I_EPOCH *
    # *************************************
    if i_epoch % eval_freq == 0:
      model.eval() # mode 'eval' model

      #progressbar = tqdm(total= test_data.num_batches(batch_size),
      #                   desc= 'Testing', leave= False)
      time_ini = dt.now()
      sumrun_loss = 0.0
      sumrun_accuracy = 0.0

      # loop through testing data in i_epoch
      check_epoch_completed = test_data.epochs_completed
      epoch_completed = test_data.epochs_completed
      i_batch = 0
      while epoch_completed == check_epoch_completed:
        # retrieve data batch
        X_batch, Y_batch = test_data.next_batch(batch_size)
        # convert data to torch format, and transfer to GPU
        X_batch, Y_batch = convert_data_torch(X_batch, Y_batch, device=_device)

        # compute model output
        pred_batch = model(X_batch)
        # compute loss
        loss = loss_fun(pred_batch, Y_batch)
        sumrun_loss += loss.item()
        # compute accuracy
        accuracy = accuracy_function(pred_batch, Y_batch, is_one_hot=False)
        sumrun_accuracy += accuracy

        #progressbar.update(1)

        i_batch += 1
        #epoch_completed = test_data.epochs_completed
        epoch_completed = -1 # train only batch data per epoch
      #end

      test_loss = sumrun_loss/i_batch
      test_accuracy = sumrun_accuracy/i_batch

      time_now = dt.now()
      print('Testing Loss = {0:1.5}. Accuracy = {1:1.5}. Time compute = {2}'
            .format(test_loss, test_accuracy, (time_now - time_ini).seconds))
    #end


    if i_epoch % 10 == 0:
      #update loss history file
      strdata = '{0} {1:1.5} {2:1.5} {3:1.5} {4:1.5}\n'.format(i_epoch+1, train_loss, test_loss, train_accuracy, test_accuracy)
      fout.write(strdata)
  #end

  print('END TRAINING...\n\n')
  fout.close()


def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))


def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)
  if not os.path.exists(FLAGS.work_dir):
    os.makedirs(FLAGS.work_dir)

  # Run the training operation
  train()


if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--is_relu_activ', type=bool, default=IS_RELU_ACTIV_DEFAULT,
                      help='Is RELU activation used after hidden layers?')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--work_dir', type=str, default=WORK_DIR_DEFAULT,
                      help='Working directory to store output data')
  parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
  FLAGS, unparsed = parser.parse_known_args()

  main()