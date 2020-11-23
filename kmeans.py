# kmeans.py     #
# Joseph Patton #

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


def kmeans(num_clusters,training_data):
    ''' find cluster centers using k-means clustering '''
    # init kmeans class #
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, 
            n_init=10, random_state=0)
    # find cluster centers #
    pred_y = kmeans.fit_predict(np.stack(training_data,axis=1))
    return kmeans.cluster_centers_


def find_widths(centers,k):
    ''' calculate the width of each activation function '''
    # r = sqrt( sum(cj - ci)^2 / k ) #
    num_centers = centers.shape[0]
    widths = np.zeros(num_centers)
    for i in range(num_centers):
        # find the square distance between this point and other points #
        distances = np.sort(np.sqrt(np.sum(s_dist(centers[i],centers),axis=1)))
        # take the k smallest distances. and calculate width. #
        # do not take distance zero because that is the       #
        # distance from the point to itself                   #
        widths[i] = np.sqrt(np.sum(distances[1:k+1])) / np.sqrt(k) 
    return widths
        

def s_dist(point, other_points):
    ''' calculate the square distance between points '''
    return np.power(other_points-point,2)
