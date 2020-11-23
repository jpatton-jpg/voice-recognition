# rbf.py                        #
# Radial Basis Function Network #
# Joseph Patton                 #

import numpy as np
from kmeans import kmeans

class ActivFunc:
    ''' 
    activation function for RBF
        Args: number of dimensions (int), 
              center of each function, one per dimension, (np float array)
              width of each function(float) 
    '''
    def __init__(self,dim=1,center=0,width=1):
        self.dim    = dim
        self.center = center
        self.width  = width
    def eval(self,value=0):
        if value.shape[1] != self.dim:
            print(f"ActiveFunc: eval() error. Wrong number of value dimensions given. Given: {value.shape[1]} Desired: {self.dim}")
        else:
            tmp = np.power(value-self.center,2)
            return np.power(np.e,np.divide(-1*np.sum(tmp,axis=1),self.width))


class ActivFuncArray(ActivFunc):
    ''' 
    array of activation functions 
        Args: number of activation functions
              number of dimensions
              initial center value
              initial width
    '''
    def __init__(self, num=1, dim=1, center=0, width=1):
        self.func = np.zeros(num,dtype=ActivFunc)
        for i in range(num):
            self.func[i] = ActivFunc(dim,center,width)
    def eval_all(self,value=0):
        return np.array([thisfunc.eval(value) for thisfunc in self.func])


#def gauss(x,y,width):
#    out = ActivFuncArray(9,2,np.zeros(2),0.25)
#    out.func[0].center = [-1,-1]
#    out.func[1].center = [-1, 0]
#    out.func[2].center = [-1, 1]
#    out.func[3].center = [ 0,-1]
#    out.func[4].center = [ 0, 0]
#    out.func[5].center = [ 0, 1]
#    out.func[6].center = [ 1,-1]
#    out.func[7].center = [ 1, 0]
#    out.func[8].center = [ 1, 1]
#    return out.eval_all(np.stack([x,y],axis=1))


def init_activation_functions(training_data,data_dimensions,num_clusters):
    ''' find the centers and widths of the activation functions '''
    # find centers using kmeans clustering #
    centers = kmeans(num_clusters,training_data)
    # init array of activation functions #
    afa = ActivFuncArray(num_clusters,data_dimensions,np.zeros(data_dimensions),1) 
    for i in range(num_clusters):
        afa.func[i].center = centers[i]
        #afa.func[i].width  = width[i]
        afa.func[i].width  = 0.25
    return afa


def evaluate_activation_functions(functions,training_data):
    return functions.eval_all(np.stack(training_data,axis=1))
        

def sum_square_error(activ_funcs,desired,weights):
    actual = np.dot(weights,activ_funcs)
    return np.sum(np.power(desired-actual,2)), desired-actual


def gradient(activ_funcs,error):
    ''' get gradient '''
    return np.transpose(np.dot(activ_funcs,error))


def train_rbf(learning_rate,training_data,dim,cluster_num):
    # initialize weights vector #
    weights = np.transpose(np.zeros(cluster_num))
    # initialize activation functions #
    activ_funcs = init_activation_functions(training_data[0:dim],dim,cluster_num)
    # evaluate activation functions at all training data points #
    evaluated_funcs = evaluate_activation_functions(activ_funcs,training_data[0:dim])
    # sum squared error difference #
    sse_diff = 100
    sse_old = 0
    # train RBF iteratively until error stops changing #
    i = 0
    while abs(sse_diff) > 1e-30:
        # get sse and desired-actual values #
        sse,error = sum_square_error(evaluated_funcs,training_data[-1],weights)
        # get gradient #
        grad = gradient(evaluated_funcs,error)
        # recalculate weights #
        weights = weights + (grad * learning_rate)
        # find difference in SSE #
        sse_diff = sse_old - sse
        sse_old = sse
        # keep track of training iterations #
        i += 1
        if i%50000 == 0:
            print(i)
    # print results #
    print("sse: ",sse)
    for i in range(len(weights)):
        print(weights[i])

# input data #
with open('x.txt', 'r') as f:
    x = np.transpose(np.array([float(x.strip()) for x in f.readlines()]))
with open('y.txt', 'r') as f:
    y = np.transpose(np.array([float(x.strip()) for x in f.readlines()]))
with open('desired.txt', 'r') as f:
    desired = np.transpose(np.array([float(x.strip()) for x in f.readlines()]))

train_rbf(0.0001,np.array([x,y,desired]),2,9)
