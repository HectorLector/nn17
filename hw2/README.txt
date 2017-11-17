Exercise Sheet 2 - Linear classification
----------------------------------------

a1.)
We calculate the most likely parameters from the training data.
Then we classify the training and test data, and check the 
missclassification rate .

We can see that the it is ~3.5% for the training data
and ~2.7% for the test data. This means our algorithm generalizes well.

See output of "nn17_ex2_a1.py":

Shape of X: (564, 18)
Shape of C: (564, 1)
Shape of X_test: (282, 18)
Shape of C_test: (282, 1)
Found classes [2,3] in training data: 286
Found classes [2,3] in test data: 149
Found classes [2] in training data: 135
Found classes [3] in training data: 151
Prior prob. class 2: 0.47202797202797203
Prior prob. class 3: 0.527972027972028
Mean class 2: [[  96.59259259]
 [  45.08888889]
 [  87.59259259]
 [ 179.34814815]
 [  61.02222222]
 [   8.65185185]
 [ 177.8962963 ]
 [  38.76296296]
 [  21.31851852]
 [ 147.71851852]
 [ 195.65185185]
 [ 484.93333333]
 [ 177.00740741]
 [  69.77777778]
 [   7.54814815]
 [  16.        ]
 [ 189.86666667]
 [ 198.13333333]]
Mean class 3: [[  91.45033113]
 [  45.32450331]
 [  76.83443709]
 [ 165.9205298 ]
 [  63.56291391]
 [   7.17218543]
 [ 170.28476821]
 [  40.13245033]
 [  20.60927152]
 [ 147.01324503]
 [ 193.54304636]
 [ 450.81456954]
 [ 182.68211921]
 [  77.42384106]
 [   4.76821192]
 [   9.82781457]
 [ 187.47682119]
 [ 191.0397351 ]]
Training data Missclassification rate: 3.4965034965034967%
Test data Missclassification rate: 2.684563758389262%
-------------------------------------------------------------

a.2)
Here we run the same calculations again, but on normalized input data.
The missclassification rate does not change.
(Only if we dont normalize the test data with the training data parameters).

See output of "nn17_ex2_a2.py":

Shape of X: (564, 18)
Shape of C: (564, 1)
Shape of X_test: (282, 18)
Shape of C_test: (282, 1)
Found classes [2,3] in training data: 286
Found classes [2,3] in test data: 149
Found classes [2] in training data: 135
Found classes [3] in training data: 151
Prior prob. class 2: 0.47202797202797203
Prior prob. class 3: 0.527972027972028
Mean class 2: [[ 0.29305651]
 [-0.02095394]
 [ 0.35533824]
 [ 0.22453997]
 [-0.17930897]
 [ 0.17968473]
 [ 0.12109125]
 [-0.10084223]
 [ 0.14086302]
 [ 0.02786323]
 [ 0.03473744]
 [ 0.09859036]
 [-0.09186142]
 [-0.50487308]
 [ 0.30348923]
 [ 0.36263966]
 [ 0.19911798]
 [ 0.4613237 ]]
Mean class 3: [[-0.26200417]
 [ 0.01873366]
 [-0.3176865 ]
 [-0.20074766]
 [ 0.16030935]
 [-0.16064528]
 [-0.10826039]
 [ 0.09015696]
 [-0.12593714]
 [-0.02491083]
 [-0.03105665]
 [-0.0881437 ]
 [ 0.08212776]
 [ 0.4513766 ]
 [-0.27133143]
 [-0.32421426]
 [-0.17801939]
 [-0.41244172]]
Training data Missclassification rate on normalized: 3.4965034965034967%
Test data Missclassification rate normalized (with Training data): 2.684563758389262%
Test data Missclassification rate normalized (with Test data): 4.026845637583892%

