import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
from numpy import *    


def fillArray(C, Cn, X, Xn):
    index = 0;
    for i in range(0, len(C)):
        if C[i] in [2,3]:
            Xn[index] = X[i];
            Cn[index] = C[i];
            index += 1;


if __name__ == '__main__':

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

    print("Shape of X: {}".format(X.shape));
    print("Shape of C: {}".format(C.shape));

    number_23 = sum(tmp in [2,3] for tmp in C);
    number_23_test = sum(tmp in [2,3] for tmp in Ctst);

    print("Found classes [2,3] in training data: {}".format(number_23));
    print("Found classes [2,3] in test data: {}".format(number_23_test));

    #Init zero array with new number of rows/columns
    X_23 = zeros((number_23, X.shape[1]));
    C_23 = zeros((number_23, C.shape[1]));
    X_23_test = zeros((number_23_test, Xtst.shape[1]));
    C_23_test = zeros((number_23_test, Ctst.shape[1]));

    #print(X_23.shape);
    #print(C_23.shape);

    #Now our new arrays should all be filled with only data from classes 2 and 3
    fillArray(C, C_23, X, X_23);
    fillArray(Ctst, C_23_test, Xtst, X_23_test);




