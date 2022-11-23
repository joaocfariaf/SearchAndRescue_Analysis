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
    precision_array = np.array([float(i) for i in file_list[2].split(',')])
    recall_array = np.array([float(i) for i in file_list[4].split(',')])

    precision = precision_array[precision_array.shape[0]//2]
    recall = recall_array[recall_array.shape[0]//2]

    for k in range(4):
        ratio = (k+1)/2
        multiplicator = ratio / (39700/29050)

        precision = 1 / (1 + multiplicator*((1/precision) -1))

        f_beta_list =[]
        beta2_array = np.arange(0, 200, 1) / 200
        for beta2 in beta2_array:    
            f_beta = (beta2 + 1)*precision*recall/(beta2*precision + recall)
            f_beta_list.append(f_beta)

        area = (2*np.sum(f_beta_list) - f_beta_list[0] - f_beta_list[199])/(2*199) 

        label = 'model; A\u03B2 = {:.3f}'
        dummy_label = 'dummy; A\u03B2 = {:.3f}'
        plt.figure()
        plt.rc('axes', labelsize=12)
        plt.rc('legend', fontsize=15)
        plt.plot(beta2_array, np.asarray(f_beta_list), label=label.format(area))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('$\u03B2 ^{2}$')
        plt.ylabel('$F_\u03B2$')
        plt.legend(loc='best')               
        plt.savefig(os.path.join(output, "f_beta_oversampling_precision_"+ str(ratio) + '_'+ model_name + '.png'))
        plt.close

    return

# fbeta('chollet', 16, 70, 'rmsprop')
# fbeta('chollet', 32, 70, 'rmsprop')
# fbeta('chollet', 128, 70, 'rmsprop')
fbeta('chollet', 256, 70, 'rmsprop')

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