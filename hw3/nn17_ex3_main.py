#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import tensorflow as tf


# plt.close('all')  # if you like


# load dataset

with open('isolet_crop_train.pkl', 'rb') as f:
    train_data = pkl.load(f)

with open('isolet_crop_test.pkl', 'rb') as f:
    test_data = pkl.load(f)

X_train, y_train = train_data
X_test, y_test = test_data


# normalize the data and check the results

# ...

print(X_train.mean(axis=0))
print(X_train.var(axis=0))


# split the data sets, etc.

# ...
