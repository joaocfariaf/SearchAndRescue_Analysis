import os
import numpy as np
import matplotlib as plt


def a_beta(precision_array, recall_array, n_thresholds=3):
    # Here we make it necessarily symmetric
    if n_thresholds%2:
        n_thresholds +=1

    precision = precision_array[precision_array.shape[0]//2]
    recall = recall_array[recall_array.shape[0]//2]

    f_beta_list =[]
    # Precision
    beta2_array = np.arange(0, 200, 1) / 200
    # Recall
    beta2_array = 200 / np.arange(1, 201, 1)
    for beta2 in beta2_array:    
        f_beta = (beta2 + 1)*precision*recall/(beta2*precision + recall)
        f_beta_list.append(f_beta)

    # Calculates the area using a simple Riemann sum
    area = (2*np.sum(f_beta_list) - f_beta_list[0] - f_beta_list[199])/(2*199) 

    label = 'model; A\u03B2 = {:.3f}'
    plt.figure()
    plt.rc('axes', labelsize=12)
    plt.rc('legend', fontsize=15) 
    plt.plot(1 / beta2_array, np.asarray(f_beta_list), label=label.format(area))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('$1/\u03B2 ^{2}$')
    plt.ylabel('$F_\u03B2$')
    plt.legend(loc='best')               
    plt.gca().invert_xaxis()  
    plt.close
