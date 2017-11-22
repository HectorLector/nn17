import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np

def number_of_samples_for_classes(X, c):
    return sum(tmp in c for tmp in X);

pl.close('all')   # closes all previous figures

# Load dataset
file_in = open('vehicle.pkl','rb')
vehicle_data = pckl.load(file_in)
file_in.close()

X = vehicle_data['train']['X'] # features
C = vehicle_data['train']['C'] # classes

indices = np.where(C == [2,3])[0] # filter class 2 and 3

input_data = X[indices]  #input vectors for class 2 and 3
output_data = C[indices] # output vectors for class 2 and 3

normalized_input = (input_data - input_data.mean(axis=0)) / input_data.std(axis=0) # normalize input features

input_data = np.insert(normalized_input, 0, 1, axis=1) # add 1 at the beginning of all input vectors for bias

output_data[np.where(output_data == 2)] = 0   # set class value for SAAB to zero
output_data[np.where(output_data == 3)] = 1   # set class value for BUS to one

theta = np.random.uniform(-0.5e-5 , 0.5e-5, (1,19))

n_epochs = 100

learning_rate = 0.01

accuracy = []
cee = []

def create_diagonal(y):
  diag = np.zeros((len(y), len(y)), dtype=np.float)
  for i in range(len(y)):
    diag[i][i] = y[i] * (1 - y[i])
  return diag

def calculate_reweighted_least_squares(input_data, theta, output_data):
  predicted, real_predicted_values = cost_function(input_data, theta)
  diff = predicted - output_data
  diag = create_diagonal(real_predicted_values)
  x_r = np.dot(input_data.T, diag)
  x_r_x = np.dot(x_r, input_data)
  x_r_x_inv = np.linalg.pinv(x_r_x)
  x_r_x_inv_x = np.dot(x_r_x_inv, input_data.T)
  return np.dot(x_r_x_inv_x, diff)



def cost_function(input_data, theta):
  product = np.dot(theta, np.transpose(input_data))
  real_predicted_values = sigmoid(np.transpose(product))
  return [(real_predicted_values >= 0.5).astype(int), real_predicted_values]

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def cross_enropy_error(predicted, output_data):
  error = 0
  predicted = np.clip(predicted, 1e-7, 1.0 - 1e-7) # clip values to avoid log(0)
  for i in range(len(output_data)):
    if output_data[i] == 0:
      error = error + (-1 * np.log(1- predicted[i]))
    else:
      error = error + (-1 * np.log(predicted[i]))
  return error


for i in range(n_epochs):
  diff = calculate_reweighted_least_squares(input_data, theta, output_data)
  theta -= diff.T

  predicted, real_predicted_values = cost_function(input_data, theta)
  epoch_accuracy = (float(len(np.where(predicted == output_data)[0]))/float(len(output_data)))
  if i>1 and epoch_accuracy == accuracy[-1] == accuracy[-2]: # stop if three consecutive accuracies are the same
    n_epochs = i
    break
  accuracy.append(epoch_accuracy)
  cee.append(cross_enropy_error(real_predicted_values, output_data))


epochs = np.arange(0, n_epochs)
pl.figure(1)
pl.plot(epochs, accuracy)
pl.xlabel('epochs')
pl.ylabel('Accuracy')
pl.title('Iterative Reweighted Least Squares')

pl.figure(2)
pl.plot(epochs, cee)
pl.xlabel('epochs')
pl.ylabel('Cross Entropy Error')
pl.title('Iterative Reweighted Least Squares')
pl.show()

