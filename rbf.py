# rbf.py                        #
# Radial Basis Function Network #
# Joseph Patton                 #

import numpy as np
from numpy import e as e
from numpy import divide as div


class ActivFunc:
    ''' activation function for RBF.
    Args: dimension (int), 
          center (one per dimension, np float array)),
          width (float)
    '''
    def __init__(self,dim=1,center=0,width=1):
        self.dim    = dim
        self.center = center
        self.width  = width
    def eval(self,value=0):
        if value.shape[1] != self.dim:
            print("ActiveFunc: eval() error. Wrong number of value dimensions given")
        else:
            tmp = np.power(value-self.center,2)
            return np.power(e,div(-1*np.sum(tmp,axis=1),self.width))


class ActivFuncArray(ActivFunc):
    ''' array of activation functions '''
    def __init__(self, num=1, dim=1, center=0, width=1):
        self.func = np.zeros(num,dtype=ActivFunc)
        for i in range(num):
            self.func[i] = ActivFunc(dim,center,width)
    def eval_all(self,value=0):
        return np.array([thisfunc.eval(value) for thisfunc in self.func])


def gauss(x,y,width):
    out = ActivFuncArray(9,2,np.zeros(2),0.25)
    out.func[0].center = [-1.,-1.]
    out.func[1].center = [-1., 0.]
    out.func[2].center = [-1., 1.]
    out.func[3].center = [ 0.,-1.]
    out.func[4].center = [ 0., 0.]
    out.func[5].center = [ 0., 1.]
    out.func[6].center = [ 1.,-1.]
    out.func[7].center = [ 1., 0.]
    out.func[8].center = [ 1., 1.]
    return out.eval_all(np.stack([x,y],axis=1))

def sum_square_error(g,desired,weights):
    actual = np.dot(weights,g)
    return np.sum(np.power(desired-actual,2)), desired-actual

def gradient(g,error):
    return np.dot(g,error)


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
    sse,error = sum_square_error(g,desired,weights)

    # get gradient #
    grad = np.transpose(gradient(g,error))

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
