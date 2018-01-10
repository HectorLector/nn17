#!/usr/bin/env python3

import numpy as np
import pickle as pkl

# load data

with open('data_train.pkl', 'rb') as f:
    train_data = pkl.load(f)

X_train, y_train = train_data

with open('data_test.pkl', 'rb') as f:
    test_data = pkl.load(f)

X_test = test_data

# your submission must have the proper format, i.e. it must be a numpy array
# with shape (337, 2), otherwise we cannot rank your submission

y_pred = np.zeros((337, 2))

# write your data to a pickle like this:

assert(type(y_pred) is np.ndarray)
assert(y_pred.shape == (337, 2))

with open('lastname1_lastname2.pkl', 'wb') as f:
    pkl.dump(y_pred, f)

