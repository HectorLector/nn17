import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
from numpy import *    

def number_of_samples_for_classes(X, c):
    return sum(tmp in c for tmp in X);

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
    print("Shape of X_test: {}".format(Xtst.shape));
    print("Shape of C_test: {}".format(Ctst.shape));

    number_23 = number_of_samples_for_classes(C, [2,3]);
    number_23_test = number_of_samples_for_classes(Ctst, [2,3]);

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

    number_2 = number_of_samples_for_classes(C_23, [2]);
    number_3 = number_of_samples_for_classes(C_23, [3]);

    print("Found classes [2] in training data: {}".format(number_2));
    print("Found classes [3] in training data: {}".format(number_3));

    p_2 = number_2 / number_23;
    p_3 = number_3 / number_23;

    print("Prior prob. class 2: {}".format(p_2));
    print("Prior prob. class 3: {}".format(p_3));

    X_2 = take(X_23, [index for index, x in enumerate(X_23) if C_23[index] == 2], axis=0);
    X_3 = take(X_23, [index for index, x in enumerate(X_23) if C_23[index] == 3], axis=0);

    print(X_2.shape);
    print(X_3.shape);

    u_2 = mean(X_2);
    u_3 = mean(X_3);

    print("Mean class 2: {}".format(u_2));
    print("Mean class 3: {}".format(u_3));

    cov_matrix = cov(X_23, rowvar=False);
    
    print(cov_matrix.shape);
    print(cov_matrix);

