import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
from numpy import *    

pl.close('all')   # closes all previous figures

# Load dataset
file_in = open('vehicle.pkl','rb')
vehicle_data = pckl.load(file_in)
file_in.close()

# Training set
X = vehicle_data['train']['X']  # features; X[i,j]...feature j of example i
C = vehicle_data['train']['C']  # classes; C[i]...class of example i
# Test set
Xtst = vehicle_data['test']['X']  # features
Ctst = vehicle_data['test']['C']  # classes



  
  
