# kmeans.py     #
# Joseph Patton #

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


def kmeans(num_clusters,training_data):
    # init kmeans class #
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, 
            n_init=10, random_state=0)
    # find cluster centers #
    pred_y = kmeans.fit_predict(np.stack(training_data,axis=1))
    return kmeans.cluster_centers_
