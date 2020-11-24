# main.py       #
# Joseph Patton #

from components import get_data
from rbf import train_rbf
from rbf import evaluate_point
import numpy as np


###################################################################
#
# Open audio file and extract training data from file
#
###################################################################

mp3_file = 'obama.mp3'
isthisobama = 1  # this audio is obama
training_data = get_data(mp3_file,isthisobama)

mp3_file = 'not_obama.mp3'
isthisobama = 0  # this audio is not obama
training_data = np.vstack((training_data,get_data(mp3_file,isthisobama)))


###################################################################
#
# Train RBF with training data
#
###################################################################

learning_rate   = 0.0001
data_dimensions = 4
cluster_num     = 10
weights,afa = train_rbf(learning_rate,training_data,data_dimensions,cluster_num)


###################################################################
#
# Test RBF with found weights and activation function parameters
#
###################################################################

success = 0
failure = 0
for data in training_data:
    val = evaluate_point(afa,weights,data[0:data_dimensions])
    if (val >= 0.5) and (data[-1] == 1):
        success += 1
    elif (val < 0.5) and (data[-1] == 0):
        success += 1
    else:
        failure += 1
print(f'Success rate: {success/(success+failure)}%')
