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
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/train-autoencode-alexnet_img_nir_2.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/snapshots/_iter_50000.caffemodel'

net = caffe.Net(model_def,
                model_weights,           # defines the structure of the model,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

net.forward()
    # The parameters are a list of [weights, biases]
data =np.copy(net.params['conv1'][0].data)
print data.shape
data = data.transpose(0, 2, 3, 1)
print data.shape
    # normalize data for display
data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
n = int(np.ceil(np.sqrt(data.shape[0])))
padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # Plot figure
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(data)


    # Save plot if filename is set

plt.savefig('/home/lod/master-thesis/python-scripts/filters.png', bbox_inches='tight', pad_inches=0)

