import fnmatch
import os
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras
import sys
import time

# Usar esse comando no terminal para testar o funcionamento:
# python3 sliding_window.py heridal heridal chollet_b64_e1_rmsprop_0.01

def sliding_window(model, image, threshold = .5, width=81, height=81, pad_x=81, pad_y=81):
    label = []
    total_width, total_height = image.size
    
    for top in range(0, (total_height - height + pad_y), pad_y):
        bottom = top + height
        for left in range(0, (total_width - width + pad_x), pad_x):
            right = left + width

            im1 = np.asarray([np.asarray(image.crop((left, top, right, bottom)))])/255
            
            imgm = image.copy()
            rect = ImageDraw.Draw(imgm)
            shape = [(left, top), (right, bottom)]

            prediction = model.predict(im1)
            if prediction >= threshold:
                x = (left + width/2) / total_width
                y = (top + height/2) / total_height
                r_width = width / total_width
                r_height = height / total_height
                
                label_line = '0 ' + str(x) + ' ' + str(y) + ' ' + str(r_width) + ' ' + str(r_height)
                label.append(label_line)


    return label

# Reads the inputs
data_set = str(sys.argv[1])
model_data_set = str(sys.argv[2])
model_name = str(sys.argv[3])


dataset_dir = os.path.join('..', '..', 'full_images', data_set)
if not os.path.exists(dataset_dir):
    raise TypeError("There's no dataset" + data_dir + " .\n")
if model_data_set == None:
    model_data_set = 'heridal'
if model_name == None:
    model_name = 'chollet_b64_e1_rmsprop_0.01'
model_dir = os.path.join('..', 'outputs', model_data_set, model_name, model_name + '.h5')
if not os.path.exists(dataset_dir):
    raise TypeError("There's no model" + model_dir + " .\n")


files = os.listdir(dataset_dir)
images = fnmatch.filter(files, "*.JPG")
start_time = time.time()
for image in images:
    im = Image.open(os.path.join(dataset_dir, image))
    model = keras.models.load_model(model_dir)
    label = sliding_window(model, im)
    label_file = os.path.join(dataset_dir, image[0:-4] + '.txt')
    file = open(label_file, "w")
    for line in label:
      file.write(line+'\n')
    file.close
end_time = time.time()
total_time = end_time - start_time
file = open(os.path.join(dataset_dir, '00-time.txt'), 'w')
file.write('Total time: ' + str(total_time) + ' s \n' + 'Num Figs: ' + str(len(images)))
file.close()
