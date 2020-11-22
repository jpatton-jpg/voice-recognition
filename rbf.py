# rbf.py                        #
# Radial Basis Function Network #
# Joseph Patton                 #

import numpy as np
from numpy import power as pw
from numpy import e as e
from numpy import divide as div

class ActivFunc:
    ''' activation function for RBF.
    Args: dimension (int), center (one per dimension, np float array)) and width (float)
    '''
    def __init__(self,dim=1,center=0,width=1):
        self.dim = dim
        self.center = center
        self.width = width
    def eval(self,value=0):
        if value.size != self.dim:
            print("ActiveFunc: eval() error. Wrong number of value dimensions given")
        else:
            tmp = pw(value - self.center,2)
            return pw(e,div(-1*np.sum(tmp),self.width))


NN = ActivFunc(2,np.array([-1,-1]),0.25)
print(NN)
print(NN.eval(np.array([-1,-1])))

'''
def gauss(x,y,width):
    out = (NN(x,y,width),NZ(x,y,width),NP(x,y,width),ZN(x,y,width),
           ZZ(x,y,width),ZP(x,y,width),PN(x,y,width),PZ(x,y,width),
           PP(x,y,width))
    return np.stack(out,axis=1)

def sum_square_error(g,desired,weights):
    actual = np.dot(g,weights)
    return np.sum(pw(desired-actual,2)), desired-actual

def gradient(g,des_min_act):
    return np.dot(des_min_act,g)


# input data #
with open('x.txt', 'r') as f:
    x = np.transpose(np.array([float(x.strip()) for x in f.readlines()]))
with open('y.txt', 'r') as f:
    y = np.transpose(np.array([float(x.strip()) for x in f.readlines()]))
with open('desired.txt', 'r') as f:
    desired = np.transpose(np.array([float(x.strip()) for x in f.readlines()]))

# RBF parameters #
width = 0.25
learning_rate = 0.01

# initialize weights vector #
weights = np.transpose(np.array([0,0,0,0,0,0,0,0,0]))

# get gaussian values #
g = gauss(x,y,width)
sse_diff = 100
sse_old = 0

# train RBF iteratively until error stops changing #
i = 0
while abs(sse_diff) > 1e-30:
    i += 1
    # get sse and desired-actual values #
    sse,des_min_act = sum_square_error(g,desired,weights)

    # get gradient #
    grad = np.transpose(gradient(g,des_min_act))

    # recalculate weights #
    weights = weights + (grad * learning_rate)

    # find difference in SSE #
    sse_diff = sse_old - sse
    sse_old = sse

    # keep track of training iterations #
    if i%50000 == 0:
        print(i)

# print results #
print("sse: ",sse)
for i in range(len(weights)):
    print(weights[i])
'''
