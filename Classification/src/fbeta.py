import fnmatch
import os
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras
import sys
import time
import math
from matplotlib import pyplot as plt
# Usar esse comando no terminal para testar o funcionamento:
# python3 sliding_window.py heridal heridal chollet_b64_e1_rmsprop_0.01

def fbeta(beta2, network_model, batch_size, n_epochs, optimizer, data_set='heridal', train_prop=6, val_prop=2, test_prop=2):
    
    output = os.path.join('..', 'outputs', data_set, 'split_'+str(train_prop)+'_'+str(val_prop)+'_'+str(test_prop))
    model_name = network_model + '_' + 'b' +  str(batch_size)  + '_' + 'e' + str(n_epochs) + '_' + optimizer
    output = os.path.join(output, model_name)

    pred_file = os.path.join(output, "pred"+model_name+".txt")
    file = open(pred_file, 'r')
    file_string = file.read()
    file_list = file_string.replace('\n', '').replace(' ', '').replace('[', '--').replace(']', '--').split('--')
    precision = np.array([float(i) for i in file_list[2].split(',')])
    recall = np.array([float(i) for i in file_list[4].split(',')])

    f_beta = (beta2 + 1)*precision*recall/(beta2*precision + recall)
    for i in f_beta:
        if math.isnan(i):
            i = 0
    threshold_array = np.arange(0, f_beta.shape[0], 1) / f_beta.shape[0]
    xmax = threshold_array[np.argmax(f_beta)]
    ymax = f_beta.max()
    plt.figure()
    plt.plot([xmax, xmax], [0.94, 1], 'k--', label="$x_{opt} =$ " + str(xmax))
    plt.plot([1/200, 1], [ymax, ymax], 'k--', label="$F_\u03B2 max =$ " + "{:.2f}".format(ymax*100) + "%")
    plt.plot(threshold_array[1:], f_beta[1:])
    plt.xlabel('Threshold')
    plt.ylabel('$F_\u03B2$')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output, 'beta2_(' + str(beta2) + ')_' + model_name + '.png'))
    plt.close

    return [xmax, ymax]

beta2_array = np.array([1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
for beta2 in beta2_array:
    fbeta(beta2, 'chollet', 256, 5, 'SGD')