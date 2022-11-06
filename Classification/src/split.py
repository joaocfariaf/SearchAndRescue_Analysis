import os
import shutil
import random
import sys
import numpy as np

data_set = str(sys.argv[1])
split = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]

patches = os.path.join("../../general_patches", "patches_" + data_set)
positive_patches = os.path.join(patches, "positive")
negative_patches = os.path.join(patches, "negative")

n_positive = len(os.listdir(positive_patches))
n_negative = len(os.listdir(negative_patches))

data = os.path.join("../inputs", "data_" + data_set) 
train_dir = os.path.join(data, "train")
val_dir = os.path.join(data, "validation")
test_dir = os.path.join(data, "test")
data_folders = [train_dir, val_dir, test_dir]

positive = [os.path.join(train_dir, "positive"), os.path.join(val_dir, "positive"), os.path.join(test_dir, "positive")]
negative = [os.path.join(train_dir, "negative"), os.path.join(val_dir, "negative"), os.path.join(test_dir, "negative")]

dir_list = [data, *data_folders, *positive, *negative]
if os.path.exists(data):
    shutil.rmtree(data)
for directory in dir_list:
    os.mkdir(directory)

pos_list = random.sample(os.listdir(positive_patches), k=n_positive)
pos = [pos_list[0:int(split[0] * n_positive)-1],
       pos_list[int(split[0] * n_positive):int(split[0] * n_positive)+int(split[1] * n_positive)-1],
       pos_list[int(split[0] * n_positive)+int(split[1] * n_positive): n_positive-1]]
       
neg_list = random.sample(os.listdir(negative_patches), k=n_negative)
neg = [neg_list[0:int(split[0] * n_negative)-1],
       neg_list[int(split[0] * n_negative):int(split[0] * n_negative)+int(split[1] * n_negative)-1],
       neg_list[int(split[0] * n_negative)+int(split[1] * n_negative):n_negative-1]]

for i in range(3):
    for file in pos[i]:
        shutil.copy(os.path.join(positive_patches, file), positive[i])
    for file in neg[i]:
        shutil.copy(os.path.join(negative_patches, file), negative[i])

#######################################################################################################################################3

test_folders = [os.path.join(data, "test0.5"), os.path.join(data, "test1.0"), os.path.join(data, "test1.5"), os.path.join(data, "test2.0")]
positive_test = []
negative_test = []
for folder in test_folders:
    positive_test.append(os.path.join(folder, "positive"))
    negative_test.append(os.path.join(folder, "negative"))

dir_list_test = [*test_folders, *positive_test]
for directory in dir_list_test:
    os.mkdir(directory)

positive_test_patches = os.path.join(test_dir, "positive")
negative_test_patches = os.path.join(test_dir, "negative")
n_test_neg = len(os.listdir(negative_test_patches))

pos_test_list = os.listdir(positive_test_patches)
pos_test = [random.sample(pos_test_list, k=int(round(n_test_neg/np.sqrt(10), 0))), random.sample(pos_test_list, k=int(round(n_test_neg/10, 0))), 
            random.sample(pos_test_list, k=int(round(n_test_neg/np.sqrt(100), 0))), random.sample(pos_test_list, k=int(round(n_test_neg/100, 0)))]

for i in range(4):
    for file in pos_test[i]:
        shutil.copy(os.path.join(positive_test_patches, file), positive_test[i])  
    shutil.copytree(negative_test_patches, negative_test[i])