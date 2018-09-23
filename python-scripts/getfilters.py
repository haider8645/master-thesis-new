# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import cv2

caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/snapshots/snapshots-fused-28-05-2018-good-results-bn-used/good-3-conv5/train-3-conv5.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/snapshots/snapshots-fused-28-05-2018-good-results-bn-used/good-3-conv5/_iter_340000.caffemodel'

net = caffe.Net(model_def,
                model_weights,           # defines the structure of the model,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

net.forward()

padding = 4

    # The parameters are a list of [weights, biases]
data =np.copy(net.params['output'][0].data)
    # N is the total number of convolutions
N = data.shape[0]*data.shape[1]
    # Ensure the resulting image is square
filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
filter_size = data.shape[2]
    # Size of the result image including padding
result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
result = np.zeros((result_size, result_size))

    # Tile the filters into the result image
filter_x = 0
filter_y = 0
for n in range(data.shape[0]):
    for c in range(data.shape[1]):
        if filter_x == filters_per_row:
            filter_y += 1
            filter_x = 0
        for i in range(filter_size):
            for j in range(filter_size):
                result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = data[n, c, i, j]
        filter_x += 1

    # Normalize image to 0-1
min = result.min()
max = result.max()
result = (result - min) / (max - min)

    # Plot figure
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(result, cmap='gray', interpolation='nearest')


    # Save plot if filename is set

plt.savefig('/home/lod/master-thesis/python-scripts/filters.png', bbox_inches='tight', pad_inches=0)

