# rbf.py                        #
# Radial Basis Function Network #
# Joseph Patton                 #


import numpy as np
from kmeans import kmeans
from kmeans import find_widths


class ActivFunc:
    def __init__(self,dim=1,center=0,width=1):
        ''' 
        initialize activation function
            Args: number of dimensions (int), 
                  center point of each function (np float array),
                  width of each function (float) 
        '''
        self.dim    = dim
        self.center = center
        self.width  = width
    def eval(self,value=0):
        # evaluating the function at multiple values #
        if len(value.shape) != 1:
            if value.shape[1] != self.dim:
                print(f'''ActiveFunc: eval() error. Wrong number of value 
                        dimensions given. Given: {value.shape[1]} Desired: 
                        {self.dim}")''')
            else:
                tmp = np.power(value-self.center,2)
                return np.power(np.e,np.divide(-1*np.sum(tmp,axis=1),self.width))
        # evaluating the function at one value #
        else:
            tmp = np.power(value-self.center,2)
            return np.power(np.e,np.divide(-1*np.sum(tmp),self.width))


class ActivFuncArray(ActivFunc):
    def __init__(self, num=1, dim=1, center=0, width=1):
        ''' 
        initialize array of activation functions 
            Args: number of activation functions
                  number of data dimensions
                  initial center point
                  initial width value
        '''
        self.func = np.zeros(num,dtype=ActivFunc)
        for i in range(num):
            self.func[i] = ActivFunc(dim,center,width)
    def eval_all(self,value=0):
        # evaluate all of the activation functions #
        return np.array([thisfunc.eval(value) for thisfunc in self.func])


def init_activation_functions(training_data,data_dimensions,num_clusters):
    ''' find the centers and widths of the activation functions '''
    print('Finding activation function values... ')
    # find centers using kmeans clustering #
    centers = kmeans(num_clusters,np.transpose(training_data))
    # init array of activation functions #
    afa=ActivFuncArray(num_clusters,data_dimensions,np.zeros(data_dimensions),1)
    # find widths #
    widths = find_widths(centers,k=2)
    for i in range(num_clusters):
        afa.func[i].center = centers[i]
        afa.func[i].width  = widths[i]
    print(f'Function Centers: {centers}')
    print(f'Function Widths: {widths}')
    return afa


def evaluate_activation_functions(functions,training_data):
    ''' evaluate the activation functions with the training data
    to train the network '''
    return functions.eval_all(training_data)


def evaluate_point(functions,weights,point):
    ''' evaluate the initialized RBF NN at a point using calculated
    weights '''
    return np.sum(functions.eval_all(point)*weights)
        

def sum_square_error(activ_funcs,desired,weights):
    ''' compute sum square error of the network '''
    actual = np.dot(weights,activ_funcs)
    return np.sum(np.power(desired-actual,2)), desired-actual


def gradient(activ_funcs,error):
    ''' get gradient '''
    return np.transpose(np.dot(activ_funcs,error))


def train_rbf(learning_rate,training_data,dim,cluster_num):
    print('Training RBF...')
    # initialize weights vector #
    weights = np.transpose(np.random.rand(cluster_num)-np.random.rand(cluster_num))
    print(f'Initial Random Weights: {weights}')
    # initialize activation functions #
    activ_funcs = init_activation_functions(training_data[:,0:dim],
                                            dim,cluster_num)
    # evaluate activation functions at all training data points #
    evaluated_funcs = evaluate_activation_functions(activ_funcs,
                                                    training_data[:,0:dim])
    # sum squared error difference #
    sse_diff = 1
    sse_old = 1000000
    # train RBF iteratively until error stops changing #
    i = 0
    while abs(sse_diff) > 1e-30:
        # get sse and desired-actual values #
        sse,error=sum_square_error(evaluated_funcs,training_data[:,-1],weights)
        # get gradient #
        grad = gradient(evaluated_funcs,error)
        # recalculate weights #
        weights = weights + (grad * learning_rate)
        # find difference in SSE #
        sse_diff = sse_old - sse
        sse_old = sse
        # keep track of training iterations #
        if i%50000 == 0 or i == 0:
            print(f'Iterations Completed: {i}   SSE: {sse}')
        i += 1
    # print results #
    print(f"Final SSE: {sse}")
    print(f'Weights: {weights}')
    return weights,activ_funcs
