'''
This archive plots the F_beta for different values of beta and considering different
values of threshold for classificating positive and negative.
This file was not considerd in the work.
'''

import os
import numpy as np
import math
from matplotlib import pyplot as plt


def fbeta(network_model, batch_size, n_epochs, optimizer, data_set='heridal', train_prop=6, val_prop=2, test_prop=2):
    
    output = os.path.join('..', 'outputs', data_set, 'split_'+str(train_prop)+'_'+str(val_prop)+'_'+str(test_prop))
    model_name = network_model + '_' + 'b' +  str(batch_size)  + '_' + 'e' + str(n_epochs) + '_' + optimizer
    output = os.path.join(output, model_name)

    pred_file = os.path.join(output, "pred"+model_name+".txt")
    file = open(pred_file, 'r')
    file_string = file.read()
    file_list = file_string.replace('\n', '').replace(' ', '').replace('[', '--').replace(']', '--').split('--')
    precision = np.array([float(i) for i in file_list[2].split(',')])
    recall = np.array([float(i) for i in file_list[4].split(',')])

    f_beta_list = []
    f_beta_med = []
    beta2_array = np.array([1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    for beta2 in beta2_array:    
        f_beta = (beta2 + 1)*precision*recall/(beta2*precision + recall)
        for i in range(f_beta.size):
            if math.isnan(f_beta[i]):
                f_beta[i] = 0
        threshold_array = np.arange(0, f_beta.shape[0], 1) / f_beta.shape[0]
        xmax = threshold_array[np.argmax(f_beta)]
        ymax = f_beta.max()
        ymed = f_beta[f_beta.shape[0]//2]
        plt.figure()
        plt.plot([xmax, xmax], [0.94, 1], 'k--', label="$x_{opt} =$ " + str(xmax))
        plt.plot([1/200, 1], [ymax, ymax], 'k--', label="$F_\u03B2 max =$ " + "{:.2f}".format(ymax*100) + "%")
        plt.plot(threshold_array[1:], f_beta[1:])
        plt.xlabel('Threshold')
        plt.ylabel('$F_\u03B2$')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(output, 'beta2_(' + str(beta2) + ')_' + model_name + '.png'))
        plt.close

        f_beta_list.append(ymax)
        f_beta_med.append(ymed)

    plt.figure()
    plt.plot(beta2_array, np.asarray(f_beta_list))
    plt.xlabel('$\u03B2 ^{2}$')
    plt.ylabel('$F_\u03B2 MAX$')                
    plt.savefig(os.path.join(output, 'beta2_f_beta_' + model_name + '.png'))
    plt.close

    ###################################################################################

    for k in range(4):
        pred_file = os.path.join(output, "test"+str((k+1)/2)+"_pred"+model_name+".txt")
        file = open(pred_file, 'r')
        file_string = file.read()
        file_list = file_string.replace('\n', '').replace(' ', '').replace('[', '--').replace(']', '--').split('--')
        precision = np.array([float(i) for i in file_list[2].split(',')])
        recall = np.array([float(i) for i in file_list[4].split(',')])

        f_beta_list = []
        f_beta_med = []
        beta2_array = np.array([1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        for beta2 in beta2_array:    
            f_beta = (beta2 + 1)*precision*recall/(beta2*precision + recall)
            for i in range(f_beta.size):
                if math.isnan(f_beta[i]):
                    f_beta[i] = 0
            threshold_array = np.arange(0, f_beta.shape[0], 1) / f_beta.shape[0]
            xmax = threshold_array[np.argmax(f_beta)]
            ymax = f_beta.max()
            ymed = f_beta[f_beta.shape[0]//2]
            plt.figure()
            plt.plot([xmax, xmax], [0.94, 1], 'k--', label="$x_{opt} =$ " + str(xmax))
            plt.plot([1/200, 1], [ymax, ymax], 'k--', label="$F_\u03B2 max =$ " + "{:.2f}".format(ymax*100) + "%")
            plt.plot(threshold_array[1:], f_beta[1:])
            plt.xlabel('Threshold')
            plt.ylabel('$F_\u03B2$')
            plt.title("test"+str((k+1)/2))
            plt.legend(loc='lower left')
            plt.savefig(os.path.join(output, "test"+str((k+1)/2)+'beta2_(' + str(beta2) + ')_' + model_name + '.png'))
            plt.close

            f_beta_list.append(ymax)
            f_beta_med.append(ymed)

        plt.figure()
        plt.plot(beta2_array, np.asarray(f_beta_list))
        plt.xlabel('$\u03B2 ^{2}$')
        plt.ylabel('$F_\u03B2 MAX$')                
        plt.savefig(os.path.join(output, "test"+str((k+1)/2)+'beta2_f_beta_' + model_name + '.png'))
        plt.close


        plt.figure()
        plt.plot(beta2_array, np.asarray(f_beta_med))
        plt.xlabel('$\u03B2 ^{2}$')
        plt.ylabel('$F_\u03B2 MAX$')                
        plt.savefig(os.path.join(output, "f_beta_med_test"+str((k+1)/2)+'beta2_f_beta_' + model_name + '.png'))
        plt.close

    return

# fbeta('chollet', 16, 70, 'rmsprop')
# fbeta('chollet', 32, 70, 'rmsprop')
fbeta('chollet', 128, 70, 'rmsprop')
# fbeta('chollet', 256, 70, 'rmsprop')

# fbeta('chollet', 16, 70, 'SGD')
# fbeta('chollet', 32, 70, 'SGD')
# fbeta('chollet', 128, 70, 'SGD')
# fbeta('chollet', 256, 70, 'SGD')

# fbeta('vasic_papic', 16, 70, 'rmsprop')
# fbeta('vasic_papic', 32, 70, 'rmsprop')
# fbeta('vasic_papic', 128, 70, 'rmsprop')
fbeta('vasic_papic', 256, 70, 'rmsprop')

# fbeta('vasic_papic', 16, 70, 'SGD')
# fbeta('vasic_papic', 32, 70, 'SGD')
# fbeta('vasic_papic', 128, 70, 'SGD')
# fbeta('vasic_papic', 256, 70, 'SGD')