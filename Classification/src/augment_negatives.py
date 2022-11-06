import os
import numpy as np
import math
from matplotlib import pyplot as plt



def reduce_positives(proportion, network_model, batch_size, n_epochs, optimizer, data_set='heridal', train_prop=6, val_prop=2, test_prop=2):
    
    data = os.path.join("../inputs", "data_" + data_set)
    test_dir = os.path.join(data, "test")

    output = os.path.join('..', 'outputs', data_set, 'split_'+str(train_prop)+'_'+str(val_prop)+'_'+str(test_prop))
    model_name = network_model + '_' + 'b' +  str(batch_size)  + '_' + 'e' + str(n_epochs) + '_' + optimizer
    output = os.path.join(output, model_name)

    
    
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    test_generator = test_datagen.flow_from_directory(
            test_dir,  # this is the target directory
            target_size=(81, 81),  # all images will be resized to 81x81
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # make predictions on the testing images, finding the index of the
    # label with the corresponding largest predicted probability
    predIdxs = model.evaluate(test_generator, steps=(NUM_TEST_IMAGES // batch_size) + 1)

    pred_file = os.path.join(output, "pred"+model_name+".txt")
    file = open(pred_file, "w")
    file.write(str(predIdxs))
    file.close()

    # Saves the model
    model_file = os.path.join(output, model_name + ".h5")
    model.save(model_file)  # always save your model after training or during training


    precision = predIdxs[2]
    recall = predIdxs[3]
    auc_pr = predIdxs[4]
    fp = predIdxs[5]
    tn = predIdxs[6]
    fpr = fp / (fp + tn)
    auc_roc = predIdxs[7]







    return


reduce_positives('chollet', 256, 5, 'SGD')