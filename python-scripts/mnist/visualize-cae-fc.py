# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import cv2
import tsne
from sklearn.cluster import KMeans

caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/train-4-conv4-earlyfusion-4k-fc.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/snapshots/_iter_11500.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

b = np.arange(3600000,dtype=float).reshape(2000, 1800)
labels = np.arange(2000).reshape(2000)

for j in range(0,2000):
    net.forward()
    b[j] = net.blobs["img_nir/concat"].data[0]                
    labels[j] = (net.blobs["label"].data[0])
    print labels[j]
    iteration_count = 'Iteration: ' + repr(j)
    print iteration_count
Y = tsne.tsne(b, no_dims= 2, initial_dims=1800, perplexity=30.0)


fig,ax = plt.subplots()

cax=ax.scatter(Y[:, 0], Y[:, 1],20,labels, cmap ='rainbow')

plt.savefig('12-05-2018-fc-4k-img-not-normalized.png')

