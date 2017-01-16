from __future__ import print_function
import numpy as np
import h5py
import os


import pandas as pd

data = np.genfromtxt('input.txt',delimiter=',')

# number of observations or number of images
size = data.shape[0]
# reshape the data
data = np.reshape(data, (size, 3, 32,  32))

#shuffle the dataset, make 4/5 be training and 1/5 be testing
import random
a = range(0, size)
random.shuffle(a)
test_size = int(round(size*0.2))

data_train = data[a[:test_size]]
data_test = data[a[test_size:]]

labels = pd.read_table("target.txt", delimiter=',', header=None)
parameters = np.array(labels.iloc[:, 0])
distributions = np.array(labels.iloc[:, 1], dtype=np.int8)


parameters_train = parameters[a[:test_size]]
parameters_test = parameters[a[test_size:]]
distributions_train = distributions[a[:test_size]]
distributions_test = distributions[a[test_size:]]


with h5py.File('data/train_parameters.h5', 'w') as f:
    f['data'] = data_train
    f['label'] = parameters_train


with h5py.File('data/test_parameters.h5', 'w') as f:
    f['data'] = data_test
    f['label'] = parameters_test


with h5py.File('data/train_distributions.h5', 'w') as f:
    f['data'] = data_train
    f['label'] = distributions_train


with h5py.File('data/test_distributions.h5', 'w') as f:
    f['data'] = data_test
    f['label'] = distributions_test


DIR = “directory link”
h5_train_parameters = os.path.join(DIR, 'train_parameters.h5')
h5_test_parameters = os.path.join(DIR, 'test_parameters.h5')

text_fn = os.path.join(DIR, 'train-parameters-path.txt')
with open(text_fn, 'w') as f:
    print(h5_train_parameters, file = f)

text_fn = os.path.join(DIR, 'test-parameters-path.txt')
with open(text_fn, 'w') as f:
     print(h5_test_parameters, file = f)


h5_train_distributions = os.path.join(DIR, 'train_distributions.h5')
h5_test_distributions = os.path.join(DIR, 'test_distributions.h5')

text_fn = os.path.join(DIR, 'train-distributions-path.txt')
with open(text_fn, 'w') as f:
    print(h5_train_distributions, file = f)

text_fn = os.path.join(DIR, 'test-distributions-path.txt')
with open(text_fn, 'w') as f:
     print(h5_test_distributions, file = f)


