
import matplotlib.pyplot as plt
import numpy as np
import os

EXERCISE = 3.2
COLORS = ['red', 'green', 'pink', 'brown', 'orange', 'yellow']


if EXERCISE == 1:
    WORK_DIR = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_1/'

    n_files = 7
    data1 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_dnn-100_relu.txt'), skiprows=1)
    data2 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_dnn-1000_relu.txt'), skiprows=1)
    data3 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_dnn-100-100_relu.txt'), skiprows=1)
    data4 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_dnn-1000-1000_relu.txt'), skiprows=1)
    data5 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_dnn-100-100-100_relu.txt'), skiprows=1)
    data6 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_dnn-1000-1000-1000_relu.txt'), skiprows=1)
    data7 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_dnn-100-100-100-100_relu.txt'), skiprows=1)
    list_data = [data1, data2, data3, data4, data5, data6, data7]
    list_labels = ['100', '1000', '100-100', '1000-1000', '100-100-100', '1000-1000-1000', '100-100-100-100']

    freq_comp_valid = 10
    list_data_valid = []
    for i in range(n_files):
        data_valid = list_data[i][0:-1:freq_comp_valid,:]
        data_valid = np.vstack((data_valid, list_data[i][-1]))
        list_data_valid.append(data_valid)
    #end

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))

    for i in range(n_files):
        data = list_data[i]
        data_valid = list_data_valid[i]

        axs[0].plot(data[:,0], data[:,1], color=COLORS[i], linewidth=2, label=list_labels[i])
        axs[1].plot(data_valid[:,0], data_valid[:,2], color=COLORS[i], linewidth=2, label=list_labels[i])
    #end

    axs[0].set_xlabel('epoch', fontsize=20)
    axs[0].set_ylabel('loss', fontsize=20)
    axs[0].tick_params(axis='x', labelsize=20)
    axs[0].tick_params(axis='y', labelsize=20)
    axs[0].legend(loc='best', fontsize=20)
    axs[0].set_title('Loss Train', fontsize=20)

    axs[1].set_xlabel('epoch', fontsize=20)
    axs[1].set_ylabel('loss', fontsize=20)
    axs[1].tick_params(axis='x', labelsize=20)
    axs[1].tick_params(axis='y', labelsize=20)
    axs[1].legend(loc='best', fontsize=20)
    axs[1].set_title('Loss Validation', fontsize=20)
    plt.show()


    fig, axs = plt.subplots(1, 2, figsize=(30, 15))

    for i in range(n_files):
        data = list_data[i]
        data_valid = list_data_valid[i]

        axs[0].plot(data[:,0], data[:,3], color=COLORS[i], linewidth=2, label=list_labels[i])
        axs[1].plot(data_valid[:,0], data_valid[:,4], color=COLORS[i], linewidth=2, label=list_labels[i])
    #end

    axs[0].set_ylim([0.0, 1.0])
    axs[0].set_xlabel('epoch', fontsize=20)
    axs[0].set_ylabel('loss', fontsize=20)
    axs[0].tick_params(axis='x', labelsize=20)
    axs[0].tick_params(axis='y', labelsize=20)
    axs[0].legend(loc='best', fontsize=20)
    axs[0].set_title('Accuracy Train', fontsize=20)

    axs[1].set_ylim([0.0, 1.0])
    axs[1].set_xlabel('epoch', fontsize=20)
    axs[1].set_ylabel('loss', fontsize=20)
    axs[1].tick_params(axis='x', labelsize=20)
    axs[1].tick_params(axis='y', labelsize=20)
    axs[1].legend(loc='best', fontsize=20)
    axs[1].set_title('Accuracy Validation', fontsize=20)
    plt.show()


    # PLOT BEST MODEL
    n_files = 1
    data1 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_dnn-100-100_relu.txt'), skiprows=1)

    freq_comp_valid = 10
    data_valid = data[0:-1:freq_comp_valid,:]
    data_valid = np.vstack((data_valid, data[-1]))

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))

    axs[0].plot(data[:,0], data[:,1], color='blue', linewidth=2, label='train')
    axs[0].plot(data_valid[:,0], data_valid[:,2], color='red', linewidth=3, label='valid')

    axs[1].plot(data[:,0], data[:,3], color='blue', linewidth=2, label='train')
    axs[1].plot(data_valid[:,0], data_valid[:,4], color='red', linewidth=3, label='valid')

    axs[0].set_xlabel('epoch', fontsize=20)
    axs[0].set_ylabel('loss', fontsize=20)
    axs[0].tick_params(axis='x', labelsize=20)
    axs[0].tick_params(axis='y', labelsize=20)
    axs[0].legend(loc='best', fontsize=20)
    axs[0].set_title('Loss', fontsize=20)

    axs[1].set_ylim([0.0, 1.0])
    axs[1].set_xlabel('epoch', fontsize=20)
    axs[1].set_ylabel('accuracy', fontsize=20)
    axs[1].tick_params(axis='x', labelsize=20)
    axs[1].tick_params(axis='y', labelsize=20)
    axs[1].legend(loc='best', fontsize=20)
    axs[1].set_title('Accuracy', fontsize=20)
    plt.show()



if EXERCISE == 2:
    WORK_DIR = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_2/'

    n_files = 1
    data = np.loadtxt(os.path.join(WORK_DIR, 'loss_history.txt'), skiprows=1)

    freq_comp_valid = 50
    data_valid = data[0:-1:freq_comp_valid,:]
    data_valid = np.vstack((data_valid, data[-1]))

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))

    axs[0].plot(data[:,0], data[:,1], color='blue', linewidth=2, label='train')
    axs[0].plot(data_valid[:,0], data_valid[:,2], color='red', linewidth=3, label='valid')

    axs[1].plot(data[:,0], data[:,3], color='blue', linewidth=2, label='train')
    axs[1].plot(data_valid[:,0], data_valid[:,4], color='red', linewidth=3, label='valid')

    axs[0].set_xlabel('epoch', fontsize=20)
    axs[0].set_ylabel('loss', fontsize=20)
    axs[0].tick_params(axis='x', labelsize=20)
    axs[0].tick_params(axis='y', labelsize=20)
    axs[0].legend(loc='best', fontsize=20)
    axs[0].set_title('Loss', fontsize=20)

    axs[1].set_ylim([0.0, 1.0])
    axs[1].set_xlabel('epoch', fontsize=20)
    axs[1].set_ylabel('accuracy', fontsize=20)
    axs[1].tick_params(axis='x', labelsize=20)
    axs[1].tick_params(axis='y', labelsize=20)
    axs[1].legend(loc='best', fontsize=20)
    axs[1].set_title('Accuracy', fontsize=20)
    plt.show()



if EXERCISE == 3:
    WORK_DIR = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_3/'

    n_files = 10
    data1 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T5_lr0001.txt'), skiprows=1)
    data2 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T10_lr0001.txt'), skiprows=1)
    data3 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T15_lr001.txt'), skiprows=1)
    data4 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T20_lr002.txt'), skiprows=1)
    data5 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T25_lr002.txt'), skiprows=1)
    data6 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T30_lr002.txt'), skiprows=1)
    data7 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T35_lr002.txt'), skiprows=1)
    data8 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T40_lr004.txt'), skiprows=1)
    data9 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T45_lr006.txt'), skiprows=1)
    data10 = np.loadtxt(os.path.join(WORK_DIR, 'loss_history_LSTM_T50_lr008.txt'), skiprows=1)
    list_data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]
    list_labels = ['T5', 'T10', 'T15', 'T20', 'T25', 'T30', 'T35', 'T40', 'T45', 'T50']

    freq_subsample = 5
    list_data_clean = []
    for i in range(n_files):
        data_clean = list_data[i][0:-1:freq_subsample,:]
        data_clean = np.vstack((data_clean, list_data[i][-1]))
        list_data_clean.append(data_clean)
    #end

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))

    for i in range(n_files):
        data = list_data_clean[i]
        axs[0].plot(data[:,0], data[:,1], color=COLORS[i], linewidth=3, label=list_labels[i])
        axs[1].plot(data[:,0], data[:,2], color=COLORS[i], linewidth=3, label=list_labels[i])
    #end

    axs[0].set_xlabel('epoch', fontsize=20)
    axs[0].set_ylabel('loss', fontsize=20)
    axs[0].tick_params(axis='x', labelsize=20)
    axs[0].tick_params(axis='y', labelsize=20)
    axs[0].legend(loc='best', fontsize=20)
    axs[0].set_title('Loss', fontsize=20)

    axs[1].set_ylim([0.0, 1.0])
    axs[1].set_xlabel('epoch', fontsize=20)
    axs[1].set_ylabel('accuracy', fontsize=20)
    axs[1].tick_params(axis='x', labelsize=20)
    axs[1].tick_params(axis='y', labelsize=20)
    axs[1].legend(loc='best', fontsize=20)
    axs[1].set_title('Acuracy', fontsize=20)
    plt.show()


    n_files = 2
    data1 = np.loadtxt(os.path.join(WORK_DIR, 'accuracy_values_RNN.txt'), skiprows=1)
    data2 = np.loadtxt(os.path.join(WORK_DIR, 'accuracy_values_LSTM.txt'), skiprows=1)

    plt.plot(data1[:,0], data1[:,1], color='blue', linewidth=3, marker='o', label='RNN')
    plt.plot(data2[:,0], data2[:,1], color='red', linewidth=3, marker='o', label='LSTM')

    plt.xlabel('sequence length', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.legend(loc='best', fontsize=20)
    plt.show()



if EXERCISE == 3.1 or EXERCISE == 3.2:
    if EXERCISE == 3.1:
        WORK_DIR = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_3_bonus1/'
    elif EXERCISE == 3.2:
        WORK_DIR = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_3_bonus2/'

    n_files = 1
    data = np.loadtxt(os.path.join(WORK_DIR, 'loss_history.txt'), skiprows=1)

    if EXERCISE == 3.2:
        freq_comp_valid = 100
        data_valid = data[0:-1:freq_comp_valid,:]
        data_valid = np.vstack((data_valid, data[-1]))
        data = data_valid

    fig, axs1 = plt.subplots(1, 1, figsize=(30, 15))

    axs1.plot(data[:,0], data[:,1], color='blue', linewidth=2, label='train loss')
    axs1.set_xlabel('epoch', fontsize=20)
    axs1.set_ylabel('train loss', fontsize=20, color='blue')
    axs1.tick_params(axis='x', labelsize=20)
    axs1.tick_params(axis='y', labelsize=20, labelcolor='blue')

    axs2 = axs1.twinx()

    axs2.plot(data[:,0], data[:,2], color='red', linewidth=2, label='train accu')
    axs2.set_ylim([0.0, 1.0])
    axs2.set_ylabel('train accuracy', fontsize=20, color='red')
    axs2.tick_params(axis='x', labelsize=20)
    axs2.tick_params(axis='y', labelsize=20, labelcolor='red')

    plt.title('Loss curves', fontsize=20)
    plt.show()



elif EXERCISE == 4:
    WORK_DIR = '/home/antonioguj/Exercises_Computer_Vision_by_Learning/results/assignment_4/'

    n_files = 1
    data = np.loadtxt(os.path.join(WORK_DIR,'loss_history.txt'), skiprows=1)

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))

    axs[0].plot(data[:,0], data[:,1], color='blue', linewidth=2)
    axs[1].plot(data[:,0], data[:,2], color='blue', linewidth=2)

    axs[0].set_xlabel('epoch', fontsize=20)
    axs[0].set_ylabel('discriminator loss', fontsize=20)
    axs[0].tick_params(axis='x', labelsize=20)
    axs[0].tick_params(axis='y', labelsize=20)
    axs[0].legend(loc='best', fontsize=20)

    axs[1].set_xlabel('epoch', fontsize=20)
    axs[1].set_ylabel('generator loss', fontsize=20)
    axs[1].tick_params(axis='x', labelsize=20)
    axs[1].tick_params(axis='y', labelsize=20)
    axs[1].legend(loc='best', fontsize=20)
    plt.show()