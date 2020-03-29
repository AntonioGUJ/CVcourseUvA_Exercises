"""
This module implements training and evaluation of a LSTM in PyTorch for classifying MNIST.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from bonus_part1.lstm_mnist import LSTM_MNIST, LSTM_MNIST_2layers

CODE_DIR = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/code/part2_RNN/bonus_part1/'
WORK_DIR_DEFAULT = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_3_bonus1/'

################################################################################


def accuracy_function(predictions, targets):
    predictions = F.softmax(predictions, dim=1)  #'F.log_softmax' already applied in model
    predictions = torch.argmax(predictions, dim=1)
    num_samples = predictions.shape[0]
    num_samples_equal = num_samples - torch.nonzero(predictions - targets).shape[0]
    return num_samples_equal / num_samples


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# The exercise says: flatten out input images as 1D vector and treat them as a sequence (each samples of 1 feat.)
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 100
learning_rate = 0.01
max_steps_epoch = 100
work_dir = WORK_DIR_DEFAULT

if not os.path.exists(work_dir):
    os.makedirs(work_dir)


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=os.path.join(CODE_DIR,'data/'),
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=os.path.join(CODE_DIR,'data/'),
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

#model = LSTM_MNIST(input_size, hidden_size, num_layers, num_classes, batch_size, is_relu_activ=True).to(device)
model = LSTM_MNIST_2layers(input_size, hidden_size, num_layers, num_classes, batch_size, is_relu_activ=True).to(device)
model.train()

# Loss and optimizer
#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


# file to store loss history
loss_history_file = os.path.join(work_dir, 'loss_history.txt')
fout = open(loss_history_file, 'w')
strheader = '/epoch/ /loss/ /accuracy/\n'
fout.write(strheader)


# Train the model
total_step = min(len(train_loader), max_steps_epoch)
for epoch in range(num_epochs):

    progressbar = tqdm(total= total_step,
                       desc= 'Epochs {}/{}'.format(epoch, num_epochs),
                       bar_format= '{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]')
    time_ini = time.time()
    sumrun_loss = 0.0
    sumrun_accuracy = 0.0

    for i, (images, labels) in enumerate(train_loader):
        # Here, you're expected to do 3 steps:
        # 1. Change the image from 2D matrix into a 1D vector (sequence of pixels)
        # 2. Forward pass of the input
        # 3. Backward pass and optimize
        # 4. Print the loss and accuracy at the end of the training epoch.

        # reshape images to [image_seq_length, image_input_size] = [28, 28]
        images = images.reshape(images.shape[0], sequence_length, input_size)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # compute model output
        predictions = model(images)
        # compute loss
        lossfun = criterion(predictions, labels)
        # backprop algorithm step
        lossfun.backward()
        optimizer.step()

        sumrun_loss += lossfun.item()
        # compute accuracy
        accuracy = accuracy_function(predictions, labels)
        sumrun_accuracy += accuracy

        loss_partial = sumrun_loss/(i+1)
        progressbar.set_postfix(loss='{0:1.5f}'.format(loss_partial))
        progressbar.update(1)

        if i==total_step-1:
            break
    #end

    train_loss = sumrun_loss/total_step
    train_accuracy = sumrun_accuracy/total_step

    time_now = time.time()
    print('\nTrain epoch {0}/{1}, Batch Size = {2}. Train Loss = {3:1.5}. Accuracy = {4:1.5}. Time compute = {5}'.
          format(epoch, total_step, batch_size, train_loss, train_accuracy, (time_now - time_ini)))

    # update loss history file
    strdata = '{0} {1:1.5} {2:1.5}\n'.format(epoch, train_loss, train_accuracy)
    fout.write(strdata)
#end


# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    total_step = len(test_loader)

    progressbar = tqdm(total=total_step, desc= 'Testing', leave= False)
    time_ini = time.time()
    sumrun_loss = 0.0
    sumrun_accuracy = 0.0

    for i, (images, labels) in enumerate(test_loader):

        # reshape images to [image_seq_length, image_input_size] = [28, 28]
        images = images.reshape(images.shape[0], sequence_length, input_size)
        images = images.to(device)
        labels = labels.to(device)

        # compute model output
        predictions = model(images)
        # compute loss
        loss = criterion(predictions, labels)
        sumrun_loss += loss.item()
        # compute accuracy
        accuracy = accuracy_function(predictions, labels)
        sumrun_accuracy += accuracy

        correct += accuracy * images.shape[0]
        total += images.shape[0]

        progressbar.update(1)
    #end

    test_loss = sumrun_loss/total_step
    test_accuracy = sumrun_accuracy/total_step

    time_now = time.time()
    print('Testing correct {0}/{1}. Testing Loss = {2:1.5}. Accuracy = {3:1.5}. Time compute = {4}'
          .format(correct, total, test_loss, test_accuracy, (time_now - time_ini)))
#end