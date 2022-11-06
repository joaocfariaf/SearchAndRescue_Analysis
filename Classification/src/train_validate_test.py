import os
import numpy as np
from time import time
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Precision, Recall, FalsePositives, TrueNegatives, AUC
from keras.optimizers import SGD
from tensorflow.python.keras.callbacks import TensorBoard

def train_validate_test(data_set, network_model, batch_size, n_epochs, optimizer, learning_rate=0.001):
        """
        def train_validate_test(network_model, batch_size, n_epochs, optimizer, learning_rate=0.01, split_ratio=[0.7,
        0.2, 0.1]): network_model = 'chollet' or 'vasic_papic' batch_size, n_epochs = int optimizer = 'rmsprop' or 'sgd'
        learning_rate = 0.01 // Is used only in the case of SGD
        """
        data = os.path.join("..", "inputs", "data_" + data_set) 
        train_dir = os.path.join(data, 'train')
        validation_dir = os.path.join(data, 'validation')
        test_dir = os.path.join(data, 'test')

        NUM_TRAIN_IMAGES = len(os.listdir(os.path.join(train_dir, 'negative'))) + len(os.listdir(os.path.join(train_dir, 'positive')))
        NUM_VALIDATION_IMAGES = len(os.listdir(os.path.join(validation_dir, 'negative'))) + len(os.listdir(os.path.join(validation_dir, 'positive')))
        NUM_TEST_IMAGES = len(os.listdir(os.path.join(test_dir, 'negative'))) + len(os.listdir(os.path.join(test_dir, 'positive')))

        N_TOTAL = NUM_TRAIN_IMAGES + NUM_VALIDATION_IMAGES + NUM_TEST_IMAGES   
        train_prop = int(round(NUM_TRAIN_IMAGES*10 / N_TOTAL, 0))
        val_prop = int(round(NUM_VALIDATION_IMAGES*10 /  N_TOTAL, 0))
        test_prop = int(round(NUM_TEST_IMAGES*10 / N_TOTAL, 0))

        output = os.path.join('..', 'outputs', data_set, 'split_'+str(train_prop)+'_'+str(val_prop)+'_'+str(test_prop))

        chollet = Sequential([
                # Camada com actv RELU e MAXPOOL 2x2
                Conv2D(32, 3, input_shape=(81, 81, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                # Camada com actv RELU e MAXPOOL 2x2
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                # Camada com actv RELU e MAXPOOL 2x2
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),

                # the model so far outputs 3D feature maps (height, width, features)
                Flatten(),  # this converts our 3D feature maps to 1D feature vectors
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
        ])

        vasic_papic = Sequential([
                # Camada com 32 filtros,actv RELU e MAXPOOL 3X3
                Conv2D(32, 3, input_shape=(81, 81, 3), activation='relu'),
                MaxPooling2D(pool_size=(3, 3), strides=3),
                # Camada com 32 filtros, actv RELU e MAXPOOL 3X3
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(3, 3), strides=3),
                # Camada com 64 filtros, actv RELU
                Conv2D(64, (3, 3), activation='relu'),
                Activation('relu'),
                # Camada com 64 filtros, actv RELU
                Conv2D(64, (3, 3), activation='relu'),

                # Fully Connected com 64 filtros, actv RELU
                Flatten(),
                Dense(64, activation='relu'),
                # Output layer com Softmax
                Dropout(0.5),
                Dense(1, activation='sigmoid')
        ])

        if network_model == 'chollet':
                model = chollet
        elif network_model == 'vasic_papic':
                model = vasic_papic
        else:
                raise TypeError("Not allowed model option.")

        thresholds = list(np.arange(0, 200, 1) / 200)
        metrics = ['accuracy', Precision(thresholds=thresholds), Recall(thresholds=thresholds), AUC(curve='PR'),
               FalsePositives(thresholds=thresholds), TrueNegatives(thresholds=thresholds), AUC(curve='ROC')]
        if optimizer == 'rmsprop':
                model.compile(loss='binary_crossentropy',
                        optimizer='rmsprop',
                        metrics=metrics)
        elif optimizer == 'SGD':
                model.compile(loss='binary_crossentropy',
                        optimizer=SGD(learning_rate=learning_rate),
                        metrics=metrics)
        else:
                raise TypeError("Not allowed optimizer option.")

        model_name = network_model + '_' + 'b' +  str(batch_size)  + '_' + 'e' + str(n_epochs) + '_' + optimizer
        output = os.path.join(output, model_name)
        logs = os.path.join(output, 'logs/{}')
        tensorboard = TensorBoard(log_dir=logs.format(time()))

        # Data Augmentation
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(rescale=1./255)
        # train_datagen = ImageDataGenerator(
        #         rescale=1./255,
        #         shear_range=0.2,
        #         zoom_range=0.2,
        #         horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling 
        validation_datagen = ImageDataGenerator(rescale=1./255)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
                train_dir,  # this is the target directory
                target_size=(81, 81),  # all images will be resized to 150x150
                batch_size=batch_size,
                class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

        # this is a similar generator, for validation data
        validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                target_size=(81, 81),
                batch_size=batch_size,
                class_mode='binary')

        # Fits the model and stores it
        H = model.fit(
                train_generator,
                steps_per_epoch= NUM_TRAIN_IMAGES // batch_size,
                epochs=n_epochs,
                validation_data=validation_generator,
                validation_steps= NUM_VALIDATION_IMAGES // batch_size,
                callbacks=[tensorboard])

        # Reescale test images before testing
        test_datagen = ImageDataGenerator(rescale=1./255)

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


        label = network_model + ' (area = {:.3f})'
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--', label="dummy")
        plt.plot(fpr, recall, label=label.format(auc_roc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output, 'roc_' + model_name + '.png'))
        plt.close()

        plt.figure()
        plt.plot([0, 1], [.5, .5], 'k--', label="dummy")
        plt.plot(recall, precision, label=label.format(auc_pr))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output, 'pr_' + model_name + '.png'))
        plt.close

        # Extra tests with different positive proportions
        extra_test_directories = [os.path.join(data, "test0.5"), os.path.join(data, "test1.0"), 
                                  os.path.join(data, "test1.5"), os.path.join(data, "test2.0")]
        for i in range(len(extra_test_directories)):                   
                test_generator = test_datagen.flow_from_directory(
                        extra_test_directories[i],  # this is the target directory
                        target_size=(81, 81),  # all images will be resized to 81x81
                        batch_size=batch_size,
                        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

                # make predictions on the testing images, finding the index of the
                # label with the corresponding largest predicted probability

                NUM_EXTRA_TEST_IMAGES = len(os.listdir(os.path.join(extra_test_directories[i],'positive'))) + \
                                        len(os.listdir(os.path.join(extra_test_directories[i],'negative')))

                predIdxs = model.evaluate(test_generator, steps=(NUM_EXTRA_TEST_IMAGES // batch_size) + 1)

                pred_file = os.path.join(output, "test"+str((i+1)/2)+"_pred"+model_name+".txt")
                file = open(pred_file, "w")
                file.write(str(predIdxs))
                file.close()

train_validate_test('heridal', 'chollet', 1, 70, 'rmsprop')
train_validate_test('heridal', 'chollet', 32, 70, 'rmsprop')
train_validate_test('heridal', 'chollet', 128, 70, 'rmsprop')
train_validate_test('heridal', 'chollet', 256, 70, 'rmsprop')

train_validate_test('heridal', 'chollet', 1, 70, 'SGD')
train_validate_test('heridal', 'chollet', 32, 70, 'SGD')
train_validate_test('heridal', 'chollet', 128, 70, 'SGD')
train_validate_test('heridal', 'chollet', 256, 70, 'SGD')

train_validate_test('heridal', 'vasic_papic', 1, 70, 'rmsprop')
train_validate_test('heridal', 'vasic_papic', 32, 70, 'rmsprop')
train_validate_test('heridal', 'vasic_papic', 128, 70, 'rmsprop')
train_validate_test('heridal', 'vasic_papic', 256, 70, 'rmsprop')

train_validate_test('heridal', 'vasic_papic', 1, 70, 'SGD')
train_validate_test('heridal', 'vasic_papic', 32, 70, 'SGD')
train_validate_test('heridal', 'vasic_papic', 128, 70, 'SGD')
train_validate_test('heridal', 'vasic_papic', 256, 70, 'SGD')