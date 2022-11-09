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

    f_beta_list =[]
    beta2_array = np.arange(0, 200, 1) / 200
    for beta2 in beta2_array:    
        f_beta = (beta2 + 1)*precision*recall/(beta2*precision + recall)
        f_beta_list.append(f_beta)

    area = 0.0
    for count in range(0, 199):
       area = f_beta_list[count] + f_beta_list[count+1] 
    area = area / (2)

    label = '2*AUC = {:.3f}'
    plt.figure()
    plt.plot(beta2_array, np.asarray(f_beta_list), label=label.format(area))
    plt.xlabel('$\u03B2 ^{2}$')
    plt.ylabel('$F_\u03B2$')
    plt.legend(loc='lower left')                
    plt.savefig(os.path.join(output, 'f_beta_med_beta2_f_beta_' + model_name + '.png'))
    plt.close

    ###################################################################################

    for k in range(4):
        pred_file = os.path.join(output, "test"+str((k+1)/2)+"_pred"+model_name+".txt")
        file = open(pred_file, 'r')
        file_string = file.read()
        file_list = file_string.replace('\n', '').replace(' ', '').replace('[', '--').replace(']', '--').split('--')
        precision_array = np.array([float(i) for i in file_list[2].split(',')])
        recall_array = np.array([float(i) for i in file_list[4].split(',')])

        precision = precision_array[precision_array.shape[0]//2]
        recall = recall_array[recall_array.shape[0]//2]

        f_beta_list =[]
        beta2_array = np.arange(0, 200, 1) / 200
        for beta2 in beta2_array:    
            f_beta = (beta2 + 1)*precision*recall/(beta2*precision + recall)
            f_beta_list.append(f_beta)

        area = 0.0
        for count in range(0, 199):
           area = f_beta_list[count] + f_beta_list[count+1] 
           area = area / (2)

        label = '2*AUC = {:.3f}'
        plt.figure()
        plt.plot(beta2_array, np.asarray(f_beta_list), label=label.format(area))
        plt.xlabel('$\u03B2 ^{2}$')
        plt.ylabel('$F_\u03B2$')
        plt.legend(loc='lower left')               
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