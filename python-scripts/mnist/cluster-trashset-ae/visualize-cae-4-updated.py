# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot, draw
import sys
import os
import cv2
import tsne
import pylab
import caffe

caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
caffe.set_mode_gpu()
# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/train-4-conv-updated.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/snapshots/_iter_200000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

a = np.arange(24200).reshape(10, 2420)
print a.shape
net.forward()
for j in range(10):
    X = (net.blobs["conv4"].data[0,j])
    print X

  
#Y = tsne.tsne(a, no_dims= 2, initial_dims=2420, perplexity = 30.0)
#pylab.scatter(Y[:, 0], Y[:, 1],20)
#pylab.savefig('cae-2420.png')

