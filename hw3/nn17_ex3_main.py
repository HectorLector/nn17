#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import tensorflow as tf


plt.close('all')  # if you like


def	normalize_input(input):
	return (input - input.mean(axis = 0)) / input.std(axis = 0);

def	split_data(data, result, size, offset=0):
	return data[offset:offset+size], result[offset:offset+size];

def one_of_class_x(output, x = 26):
	one_of_x = [];
	for i in range(output.shape[0]):
	    item = np.zeros((x));
	    item[output[i] - 1] = 1;
	    one_of_x.append(item);
    #one hat encoding
	return np.array(one_of_x);

def	create_batch(data_in, data_out, size=40):
    x_size = data_in.shape[0];
    nr_batches = int(x_size / size) + 1;
    data_in_batches = np.array_split(data_in, nr_batches);
    data_out_batches = np.array_split(data_out, nr_batches);
    return (data_in_batches, data_out_batches);


# load dataset

with open('isolet_crop_train.pkl', 'rb') as f:
    train_data = pkl.load(f)

with open('isolet_crop_test.pkl', 'rb') as f:
    test_data = pkl.load(f)

X_train, y_train = train_data;
X_test, y_test = test_data;


# normalize the data and check the results

print(X_train.shape);
print(y_train.shape);
print(X_test.shape);
print(y_test.shape);

X_train = normalize_input(X_train);
X_test = normalize_input(X_test);

print(X_train.mean(axis = 0));
print(X_train.var(axis = 0));


# split the data sets, etc.
training_size = X_train.shape[0];
size_70 = int(training_size * 0.7);
size_15 = int(training_size * 0.15);

#here we loose 1-2 training examples due to int rounding error
X_train_t, y_train_t = split_data(X_train, y_train, size_70, 0);
X_train_v, y_train_v = split_data(X_train, y_train, size_15, size_70);
X_train_e, y_train_e = split_data(X_train, y_train, size_15, size_70 + size_15);

print("70% training set x: ", X_train_t.shape);
print("70% training set y: ", y_train_t.shape);
print("15% validation set x: ", X_train_v.shape);
print("15% validation set y: ", y_train_v.shape);
print("15% early stopping set x: ", X_train_e.shape);
print("15% early stopping set y: ", y_train_e.shape);


# Neural network

nr_in = X_train.shape[1]; #300 features
nr_hidden = 20;
nr_out = 26;
rate = 0.001;
nr_runs = 200;

#placeholder arrays, 64bit numbers
w_0 = np.array(np.random.randn(nr_in, nr_hidden) / np.sqrt(nr_in), dtype = np.float64)
b_0 = np.zeros(nr_hidden, dtype = np.float64)
w_1 = np.array(np.random.randn(nr_hidden, nr_out) / np.sqrt(nr_hidden), dtype = np.float64)
b_1 = np.zeros(nr_out, dtype = np.float64)

#tensorflow
input_hidden_weights = tf.Variable(initial_value = w_0, trainable = True)
input_hidden_bias = tf.Variable(initial_value = b_0, trainable = True)

hidden_output_weights = tf.Variable(initial_value = w_1, trainable = True)
hidden_output_bias = tf.Variable(initial_value = b_1, trainable = True)

input_layer = tf.placeholder(tf.float64, shape = [None, nr_in])
actual_output = tf.placeholder(tf.float64, shape = [None, nr_out])


#check different activation functions for hidden layer
#y = tf.nn.sigmoid(tf.matmul(input_layer, input_hidden_weights) + input_hidden_bias)
#y = tf.nn.tanh(tf.matmul(input_layer, input_hidden_weights) + input_hidden_bias)
y = tf.nn.relu(tf.matmul(input_layer, input_hidden_weights) + input_hidden_bias)
z = tf.matmul(y, hidden_output_weights) + hidden_output_bias

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = z, labels = actual_output)) # stable softmax

# accuracy
correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(actual_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

#check different algorithms compared to "standard" gradient decent
#train_step = tf.train.GradientDescentOptimizer(rate).minimize(cross_entropy)
#train_step = tf.train.RMSPropOptimizer(rate).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)

init = tf.global_variables_initializer();
session = tf.Session();
session.run(init);

#output (t)rain, (v)alidation and (e)arly stopping sets (one hat encoded)
y_t_one_class = one_of_class_x(y_train_t);
y_v_one_class = one_of_class_x(y_train_v);
y_e_one_class = one_of_class_x(y_train_e);

train_err = [];
train_err_stop = [];
train_acc = [];
current_accuracy_max = 0;
high_accuracy_weights = [];
high_accuracy_index = 0;
early_stopping_last_index = 5;
stopped_early = False;

for epoch in range(0, nr_runs):
    #X_training, C_training_hot = shuffle_input_output_similiarily(X_training, C_training_hot)
    in_batches, out_batches = create_batch(X_train_t, y_t_one_class);
    for i in range(len(in_batches)):
        session.run(train_step, feed_dict = {input_layer: in_batches[i], actual_output: out_batches[i]});

    #validation set
    err_val = session.run(cross_entropy, feed_dict = {input_layer: X_train_v, actual_output: y_v_one_class});
    train_err.append(err_val);

    #early stopping set
    err_stopping = session.run(cross_entropy, feed_dict = {input_layer: X_train_e, actual_output: y_e_one_class});


    print("Run: {}, error validation: {:.5f}, error early stopping: {:.5f}".format(epoch, err_val, err_stopping));

    acc = session.run(accuracy, feed_dict = {input_layer: X_train_v, actual_output: y_v_one_class});
    train_acc.append(acc);
    print("Accuracy: {:.5f}".format(acc));

    if acc >= current_accuracy_max:
        current_accuracy_max = acc;
        high_accuracy_index = epoch;
        weights = [session.run(input_hidden_weights), session.run(input_hidden_bias), session.run(hidden_output_weights), session.run(hidden_output_bias)];

    #early stopping: if the max value of the last x errors is smaller than current error, stop
    if (len(train_err_stop) > 0) and (np.max(train_err_stop[-early_stopping_last_index:]) < err_stopping):
        print("Max error in the last {} iterations was smaller than current error: {:.5f} - Early Stopping engaded!".format(early_stopping_last_index, err_stopping));
        stopped_early = True;
        reak;

    train_err_stop.append(err_stopping);


plt.figure(1)
plt.subplot(211)
plt.plot(np.arange(len(train_err)), train_err, label="Error")
plt.plot(np.arange(len(train_err_stop)), train_err_stop, label="Early Stopping Error")
plt.xlabel('Epochs')
plt.ylabel('Error')

plt.subplot(212)
plt.plot(np.arange(len(train_acc)), train_acc, label="Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


