import fnmatch
import os
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras
import sys
import time

# Reads the inputs
data_set = 'heridal'#str(sys.argv[1])
model_data_set = 'heridal'#str(sys.argv[2])
# model_name = str(sys.argv[3])
dataset_dir = os.path.join('..', '..', 'full_images', data_set)
if not os.path.exists(dataset_dir):
    raise TypeError("There's no dataset" + data_dir + " .\n")
if model_data_set == None:
    model_data_set = 'heridal'


def sliding_window(model, image, threshold = .5, width=81, height=81, stride_x=80, stride_y=75):
    label = []
    total_width, total_height = image.size
    
    for top in range(0, (total_height - height + stride_y), stride_y):
        bottom = top + height
        
        img = image.crop((0, top, total_width, bottom))
        window = np.asarray([np.asarray(img)])/255

        print(window.shape)

        pred = model.predict(np.expand_dims(np.expand_dims(window, 2), 0))[0]

        print(window.shape)
 
        break

        for left in range(0, (total_width - width + stride_x), stride_x):
            right = left + width

            # Gets the window
            window = np.asarray([np.asarray(image.crop((left, top, right, bottom)))])/255
            

            prediction = model.predict(window, verbose=0)
            if prediction >= threshold:
                x = (left + width/2) / total_width
                y = (top + height/2) / total_height
                r_width = width / total_width
                r_height = height / total_height
                            
                label_line = '0 ' + str(x) + ' ' + str(y) + ' ' + str(r_width) + ' ' + str(r_height)
                label.append(label_line)
            
            
    return label


names = ['chollet_b256_e70_rmsprop', 'vasic_papic_b256_e70_rmsprop']
for model_name in names:
    model_dir = os.path.join('..', 'outputs', model_data_set, 'split_6_2_2', model_name, model_name + '.h5')
    if not os.path.exists(dataset_dir):
        raise TypeError("There's no model" + model_dir + " .\n")

    model = keras.models.load_model(model_dir)
    # Iterates over the images in the dataset


    def slide_through(image):
        im = Image.open(os.path.join(dataset_dir, image))    
        start_time = time.time()
        label = sliding_window(model, im)
        total_time = time.time() - start_time
        label_file = os.path.join(dataset_dir, image[0:-4] + '.txt')
        file = open(label_file, "w")
        for line in label:
            file.write(line+'\n')
            file.close
        return total_time


    files = os.listdir(dataset_dir)
    # images = fnmatch.filter(files, "*.JPG")
    images = fnmatch.filter(files, "test_BLI_0001.JPG")
    total_time = []

    # pool = mp.Pool(mp.cpu_count())
    # pool.map(slide_through, [image for image in images])
    # # pool.starmap(slide_through, [(image, model, dataset_dir) for image in images])
    # pool.close()
    
    for image in images:
        total_time.append(slide_through(image))

    # Calculates and stores statistical data about the time to slide an entire image
    avg_time = np.mean(np.array(total_time))
    std_time = np.mean(np.array(total_time))
    file = open(os.path.join(dataset_dir,  '00-time' + model_name + '.txt'), 'w')
    file.write('Time per Figure: '+ str(avg_time) + '  +-  ' + str(std_time) +'\n Total time: \n' + str(total_time) + ' s \n' + 'Num Figs: ' + str(len(images)))
    file.close()
    print('\n')
    print(avg_time)
    print(std_time)
    print('\n')
    break
