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

def sigmoid_log(val):
     if val > 0:    # avoid log overlfow errors
         return 1.0 / (1.0 + exp(-val));
     else:
         return exp(val) / (1.0 + exp(val));

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

    #Normalize
    mean_23 = X_23.mean(axis=0);
    std_23 = X_23.std(axis=0);
    X_23_norm = (X_23 - mean_23) / std_23;
    #normalization parameters from training set to normalize test set
    X_23_test_norm = (X_23_test - mean_23)/std_23;
    X_23_test_norm2 = (X_23_test - X_23_test.mean(axis=0)) / X_23_test.std(axis=0);

    number_2 = number_of_samples_for_classes(C_23, [2]);
    number_3 = number_of_samples_for_classes(C_23, [3]);

    print("Found classes [2] in training data: {}".format(number_2));
    print("Found classes [3] in training data: {}".format(number_3));

    p_2 = number_2 / number_23;
    p_3 = number_3 / number_23;

    print("Prior prob. class 2: {}".format(p_2));
    print("Prior prob. class 3: {}".format(p_3));

    X_2 = take(X_23_norm, [index for index, x in enumerate(X_23) if C_23[index] == 2], axis=0);
    X_3 = take(X_23_norm, [index for index, x in enumerate(X_23) if C_23[index] == 3], axis=0);

    #print(X_2.shape);
    #print(X_3.shape);

    u_2 = mean(X_2, axis=0).reshape(18,1);
    u_3 = mean(X_3, axis=0).reshape(18,1);

    print("Mean class 2: {}".format(u_2));
    print("Mean class 3: {}".format(u_3));

    cov_matrix = cov(X_23_norm, rowvar=False);

    cov_inverse = linalg.inv(cov_matrix)
    
    #print(cov_matrix);
    #print(cov_inverse.shape);
    
    u_dif = (u_2 - u_3);
    #print(u_dif.shape);
    w = cov_inverse.dot(u_dif);
    w_0 = -1/2*transpose(u_2).dot(cov_inverse).dot(u_2) + 1/2*transpose(u_3).dot(cov_inverse).dot(u_3) + math.log(p_2/p_3);

    #print(w.shape);
    #print(w_0.shape);

    wrong_prediction_count = 0;

    for i in range(0, number_23):
        x = X_23_norm[i];
        real_value = C_23[i];
        tmp = transpose(w).dot(x) + w_0;
        #print(tmp.shape);
        res = sigmoid_log(tmp);
        predicted_class = 2 if res > 0.5 else 3;
        if predicted_class != real_value: 
            #print("Propability: {} vs. Predict: {} vs. Real: {}".format(res, predicted_class, real_value));
            wrong_prediction_count += 1;
        #print("Propability: {} vs. Predict: {} vs. Real: {}".format(res, predicted_class, real_value));

    print("Training data Missclassification rate on normalized: {}%".format(wrong_prediction_count/number_23 * 100));

    wrong_prediction_count = 0;

    for i in range(0, number_23_test):
        x = X_23_test_norm[i];
        #print(x);
        real_value = C_23_test[i];
        tmp = transpose(w).dot(x) + w_0;
        #print(tmp.shape);
        res = sigmoid_log(tmp);
        predicted_class = 2 if res > 0.5 else 3;
        if predicted_class != real_value: 
            #print("Propability: {} vs. Predict: {} vs. Real: {}".format(res, predicted_class, real_value));
            wrong_prediction_count += 1;
        #print("Propability: {} vs. Predict: {} vs. Real: {}".format(res, predicted_class, real_value));

    print("Test data Missclassification rate normalized (with Training data): {}%".format(wrong_prediction_count/number_23_test * 100));

    wrong_prediction_count = 0;

    for i in range(0, number_23_test):
        x = X_23_test_norm2[i];
        #print(x);
        real_value = C_23_test[i];
        tmp = transpose(w).dot(x) + w_0;
        #print(tmp.shape);
        res = sigmoid_log(tmp);
        predicted_class = 2 if res > 0.5 else 3;
        if predicted_class != real_value: 
            #print("Propability: {} vs. Predict: {} vs. Real: {}".format(res, predicted_class, real_value));
            wrong_prediction_count += 1;
        #print("Propability: {} vs. Predict: {} vs. Real: {}".format(res, predicted_class, real_value));

    print("Test data Missclassification rate normalized (with Test data): {}%".format(wrong_prediction_count/number_23_test * 100));
